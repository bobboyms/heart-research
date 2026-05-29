#!/usr/bin/env python3
"""Convert papers to Markdown with arXiv source, static extraction, and optional LLM cleanup.

Usage:
    uv run python scripts/pdf_to_markdown.py path/to/file.pdf
    uv run python scripts/pdf_to_markdown.py --all papers --mode static
    uv run python scripts/pdf_to_markdown.py path/to/file.pdf --source require
    uv run python scripts/pdf_to_markdown.py --all papers --mode hybrid --clean all
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import mimetypes
import os
import re
import shutil
import tarfile
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
MAX_DIRECT_INPUT_BYTES = 50 * 1024 * 1024
DEFAULT_CHUNK_CHARS = 18_000
DEFAULT_SOURCE_CACHE = Path(".cache") / "arxiv-sources"
TEX_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
BIB_COMMAND_RE = re.compile(r"\\bibliography\{([^{}]+)\}")
GRAPHICS_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg")
DISPLAY_MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "flalign",
    "flalign*",
    "alignat",
    "alignat*",
    "eqnarray",
    "eqnarray*",
}


@dataclass(frozen=True)
class PageMarkdown:
    page_number: int
    text: str


@dataclass(frozen=True)
class MarkdownChunk:
    start_page: int
    end_page: int
    text: str


@dataclass(frozen=True)
class ConversionConfig:
    mode: str
    clean: str
    source: str
    source_cache: Path
    chunk_chars: int
    keep_raw: bool
    api_key: str | None
    model: str
    timeout: int


@dataclass(frozen=True)
class ConversionReport:
    pdf_path: Path
    output_path: Path
    source: str
    math_source: str
    figures: int = 0
    tables: int = 0
    assets: int = 0
    references: int = 0
    unresolved_macros: int = 0
    llm_chunks: int = 0


@dataclass(frozen=True)
class LatexConversionResult:
    markdown: str
    figures: int
    tables: int
    assets: int
    references: int
    unresolved_macros: int


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        values[key] = value

    return values


def env_value(env_file_values: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or env_file_values.get(key) or default


def send_request(request: urllib.request.Request, *, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach OpenAI API: {exc.reason}") from exc


def http_json_request(
    url: str,
    *,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return send_request(request, timeout=timeout)


def multipart_form_data(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----openai-pdf-md-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def upload_pdf(pdf_path: Path, *, api_key: str, timeout: int) -> str:
    body, boundary = multipart_form_data(
        fields={"purpose": "user_data"},
        files={"file": pdf_path},
    )
    request = urllib.request.Request(
        f"{OPENAI_BASE_URL}/files",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    result = send_request(request, timeout=timeout)

    file_id = result.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError(f"Upload succeeded but no file id was returned: {result}")

    return file_id


def delete_uploaded_file(file_id: str, *, api_key: str, timeout: int) -> None:
    request = urllib.request.Request(
        f"{OPENAI_BASE_URL}/files/{file_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        send_request(request, timeout=timeout)
    except RuntimeError:
        pass


def extract_output_text(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n\n".join(parts).strip()


def require_api_key(api_key: str | None) -> str:
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in the environment or scripts/.env")
    return api_key


def detect_arxiv_id(pdf_path: Path) -> str | None:
    match = ARXIV_ID_RE.search(pdf_path.stem)
    if match:
        return match.group(1)
    return None


def download_arxiv_source(arxiv_id: str, *, cache_dir: Path, timeout: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{arxiv_id}.src"
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return cached_path

    request = urllib.request.Request(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id),
        headers={"User-Agent": "llm-research-pdf-to-markdown/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            cached_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"arXiv returned HTTP {exc.code} for {arxiv_id}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download arXiv source for {arxiv_id}: {exc.reason}") from exc

    return cached_path


def safe_extract_tar(fileobj: io.BytesIO, target_dir: Path) -> bool:
    try:
        with tarfile.open(fileobj=fileobj, mode="r:*") as archive:
            for member in archive.getmembers():
                member_path = (target_dir / member.name).resolve()
                if not str(member_path).startswith(str(target_dir.resolve())):
                    raise RuntimeError(f"Unsafe path in arXiv source archive: {member.name}")
            try:
                archive.extractall(target_dir, filter="data")
            except TypeError:
                archive.extractall(target_dir)
        return True
    except tarfile.TarError:
        return False


def unpack_arxiv_source(source_path: Path, target_dir: Path) -> None:
    data = source_path.read_bytes()

    if safe_extract_tar(io.BytesIO(data), target_dir):
        return

    try:
        decompressed = gzip.decompress(data)
    except OSError:
        decompressed = data

    if safe_extract_tar(io.BytesIO(decompressed), target_dir):
        return

    tex_path = target_dir / f"{source_path.stem}.tex"
    tex_path.write_bytes(decompressed)


def read_text_guess(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def strip_latex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        escaped = False
        kept: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            kept.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def normalize_tex_ref(raw_ref: str, base_dir: Path) -> Path | None:
    ref = raw_ref.strip()
    if not ref:
        return None
    candidates = [base_dir / ref]
    if not ref.endswith(".tex"):
        candidates.append(base_dir / f"{ref}.tex")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def expand_tex_inputs(path: Path, *, seen: set[Path] | None = None) -> str:
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        return ""
    seen.add(resolved)

    text = strip_latex_comments(read_text_guess(resolved))

    def replace_input(match: re.Match[str]) -> str:
        input_path = normalize_tex_ref(match.group(1), resolved.parent)
        if not input_path:
            return f"\n<!-- missing TeX input: {match.group(1)} -->\n"
        return "\n" + expand_tex_inputs(input_path, seen=seen) + "\n"

    return TEX_INPUT_RE.sub(replace_input, text)


def find_main_tex(root_dir: Path) -> Path:
    tex_files = sorted(root_dir.rglob("*.tex"))
    if not tex_files:
        raise RuntimeError("No .tex files found in arXiv source.")

    scored: list[tuple[int, Path]] = []
    for path in tex_files:
        text = read_text_guess(path)
        score = 0
        if "\\documentclass" in text:
            score += 100
        if "\\begin{document}" in text:
            score += 100
        if "\\title" in text:
            score += 10
        if "\\section" in text:
            score += 10
        score += min(len(text) // 10_000, 20)
        scored.append((score, path))

    scored.sort(key=lambda item: (item[0], len(read_text_guess(item[1]))), reverse=True)
    best_score, best_path = scored[0]
    if best_score <= 0:
        raise RuntimeError("Could not identify the main .tex file in arXiv source.")
    return best_path


def extract_document_body(tex: str) -> str:
    begin = re.search(r"\\begin\{document\}", tex)
    end = re.search(r"\\end\{document\}", tex)
    if begin and end and end.start() > begin.end():
        return tex[begin.end() : end.start()]
    if begin:
        return tex[begin.end() :]
    return tex


def extract_balanced_brace(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    chars: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return None


def extract_simple_macros(tex: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    command_pattern = re.compile(
        r"\\(?:re)?newcommand\*?\s*(?:\{\\([A-Za-z]+)\}|\\([A-Za-z]+))(?:\[[^\]]+\])?\s*"
    )
    for match in command_pattern.finditer(tex):
        name = match.group(1) or match.group(2)
        value = extract_balanced_brace(tex, match.end())
        if not name or not value:
            continue
        expansion, _ = value
        if "#" in expansion:
            continue
        macros[name] = expansion.strip()

    operator_pattern = re.compile(r"\\DeclareMathOperator\*?\s*\{\\([A-Za-z]+)\}\s*\{([^{}]+)\}")
    for match in operator_pattern.finditer(tex):
        macros[match.group(1)] = rf"\operatorname{{{match.group(2).strip()}}}"

    def_pattern = re.compile(r"\\def\\([A-Za-z]+)\s*")
    for match in def_pattern.finditer(tex):
        name = match.group(1)
        value = extract_balanced_brace(tex, match.end())
        if not name or not value:
            continue
        expansion, _ = value
        if "#" in expansion:
            continue
        macros[name] = expansion.strip()

    return macros


def apply_simple_macros(tex: str, macros: dict[str, str]) -> str:
    if not macros:
        return tex

    for name in sorted(macros, key=len, reverse=True):
        expansion = macros[name]
        tex = re.sub(
            rf"\\{re.escape(name)}(?![A-Za-z])",
            lambda _match, value=expansion: value,
            tex,
        )
    return tex


def strip_math_regions(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", " ", text, flags=re.S)
    return text


def count_unresolved_text_macros(markdown: str) -> int:
    without_math = strip_math_regions(markdown)
    macros = set(re.findall(r"\\([A-Za-z]+)\*?(?![A-Za-z])", without_math))
    allowed = {"begin", "end", "label"}
    return len(macros - allowed)


def clean_latex_command_markup(text: str) -> str:
    text = text.strip()
    text = text.replace("{-}", "-")
    text = re.sub(r"\\caption(?:\[[^\]]*\])?\{(.*?)\}", r"\1", text, flags=re.S)
    text = clean_latex_markdown(convert_inline_math(text))
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def extract_first_command_arg(text: str, command: str) -> str | None:
    pattern = re.compile(rf"\\{command}(?:\[[^\]]*\])?\s*")
    match = pattern.search(text)
    if not match:
        return None
    value = extract_balanced_brace(text, match.end())
    if not value:
        return None
    return value[0]


def find_source_asset(source_dir: Path, raw_ref: str) -> Path | None:
    ref = raw_ref.strip()
    if not ref:
        return None

    candidates = [source_dir / ref]
    if Path(ref).suffix.lower() not in GRAPHICS_EXTENSIONS:
        candidates.extend(source_dir / f"{ref}{extension}" for extension in GRAPHICS_EXTENSIONS)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    basename = Path(ref).name
    for candidate in source_dir.rglob(f"{basename}*"):
        if candidate.is_file() and candidate.suffix.lower() in GRAPHICS_EXTENSIONS:
            return candidate

    return None


def copy_asset_for_markdown(asset_path: Path, output_path: Path) -> str:
    assets_dir = output_path.with_name(f"{output_path.stem}_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    destination = assets_dir / asset_path.name
    if destination.exists() and destination.read_bytes() != asset_path.read_bytes():
        stem = asset_path.stem
        suffix = asset_path.suffix
        counter = 2
        while True:
            candidate = assets_dir / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                destination = candidate
                break
            counter += 1

    if not destination.exists():
        shutil.copy2(asset_path, destination)

    return f"{assets_dir.name}/{destination.name}"


def convert_figure_environments(
    text: str,
    *,
    source_dir: Path,
    output_path: Path,
) -> tuple[str, int, int]:
    figures = 0
    assets = 0
    pattern = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)

    def replace(match: re.Match[str]) -> str:
        nonlocal figures, assets
        block = match.group(1)
        figures += 1
        caption = extract_first_command_arg(block, "caption")
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", block)

        lines: list[str] = []
        for graphic_ref in graphics:
            asset_path = find_source_asset(source_dir, graphic_ref)
            if asset_path:
                rel_path = copy_asset_for_markdown(asset_path, output_path)
                assets += 1
            else:
                rel_path = graphic_ref.strip()
            lines.append(f"![]({rel_path})")

        if caption:
            lines.extend(["", f"> Figure: {clean_latex_command_markup(caption)}"])

        if not lines:
            lines.append("<!-- Figure omitted: no includegraphics found. -->")

        return "\n\n" + "\n".join(lines).strip() + "\n\n"

    return pattern.sub(replace, text), figures, assets


def convert_table_environments(text: str) -> tuple[str, int]:
    tables = 0
    pattern = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.S)

    def replace(match: re.Match[str]) -> str:
        nonlocal tables
        block = match.group(1).strip()
        tables += 1
        caption = extract_first_command_arg(block, "caption")
        block = re.sub(r"\\caption(?:\[[^\]]*\])?\{.*?\}", "", block, flags=re.S)
        block = re.sub(r"\\label\{[^{}]*\}", "", block)
        block = block.strip()

        lines: list[str] = []
        if caption:
            lines.append(f"> Table: {clean_latex_command_markup(caption)}")
        if block:
            lines.extend(["", "```latex", block, "```"])
        return "\n\n" + "\n".join(lines).strip() + "\n\n"

    return pattern.sub(replace, text), tables


def remove_layout_commands(text: str) -> str:
    commands_with_args = [
        "vspace",
        "hspace",
        "vskip",
        "hskip",
        "resizebox",
        "scalebox",
        "setlength",
    ]
    for command in commands_with_args:
        text = re.sub(rf"\\{command}\*?(?:\[[^\]]*\])?\{{[^{{}}]*\}}", "", text)

    wrappers = [
        "center",
        "flushleft",
        "flushright",
        "CJK",
        "CJK*",
        "small",
        "scriptsize",
        "footnotesize",
        "normalsize",
        "quote",
        "quotation",
    ]
    for wrapper in wrappers:
        escaped = re.escape(wrapper)
        text = re.sub(rf"\\begin\{{{escaped}\}}(?:\{{[^{{}}]*\}})*", "", text)
        text = re.sub(rf"\\end\{{{escaped}\}}", "", text)

    text = re.sub(r"\{CJK\*?\}\{[^{}]*\}\{[^{}]*\}", "", text)
    return text


def clean_latex_text(text: str) -> str:
    text = text.replace("~", " ")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\$", "$")
    text = text.replace(r"\{", "{")
    text = text.replace(r"\}", "}")
    text = re.sub(r"\\emph\{([^{}]*)\}", r"_\1_", text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"_\1_", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\textsc\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"[\2](\1)", text)
    text = re.sub(r"\\(?:cite|citep|citet|citealp|ref|eqref|autoref)\*?(?:\[[^\]]*\])?\{([^{}]+)\}", r"[\1]", text)
    text = re.sub(r"\\label\{[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_latex_markdown(text: str) -> str:
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"@@MATH_BLOCK_{len(protected) - 1}@@"

    text = re.sub(r"\$\$.*?\$\$", stash, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", stash, text, flags=re.S)
    text = clean_latex_text(text)
    for index, value in enumerate(protected):
        text = text.replace(f"@@MATH_BLOCK_{index}@@", value)
    return text


def convert_inline_math(text: str) -> str:
    text = re.sub(r"\\\((.*?)\\\)", lambda m: f"${m.group(1).strip()}$", text, flags=re.S)
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", text, flags=re.S)
    return text


def convert_display_math_envs(text: str) -> str:
    for env in sorted(DISPLAY_MATH_ENVS, key=len, reverse=True):
        escaped_env = re.escape(env)
        pattern = re.compile(rf"\\begin\{{{escaped_env}\}}(.*?)\\end\{{{escaped_env}\}}", re.S)

        def replace(match: re.Match[str], env_name: str = env) -> str:
            body = match.group(1).strip()
            if env_name.startswith("equation"):
                return f"\n\n$$\n{body}\n$$\n\n"
            return f"\n\n$$\n\\begin{{{env_name}}}\n{body}\n\\end{{{env_name}}}\n$$\n\n"

        text = pattern.sub(replace, text)
    return text


def convert_lists(text: str) -> str:
    text = re.sub(r"\\begin\{(?:itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate)\}", "\n", text)
    text = re.sub(r"\n\s*\\item(?:\[[^\]]+\])?\s*", "\n- ", text)
    return text


def convert_sections(text: str) -> str:
    section_map = {
        "part": "#",
        "chapter": "#",
        "section": "##",
        "subsection": "###",
        "subsubsection": "####",
        "paragraph": "#####",
    }
    for command, prefix in section_map.items():
        pattern = re.compile(rf"\\{command}\*?\{{([^{{}}]+)\}}")
        text = pattern.sub(lambda m, p=prefix: f"\n\n{p} {clean_latex_text(m.group(1))}\n\n", text)
    return text


def extract_command_value(tex: str, command: str) -> str | None:
    match = re.search(rf"\\{command}\{{(.+?)\}}", tex, flags=re.S)
    if not match:
        return None
    return clean_latex_text(match.group(1))


def convert_abstract(text: str) -> str:
    return re.sub(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        lambda m: f"\n\n## Abstract\n\n{clean_latex_markdown(convert_inline_math(m.group(1)))}\n\n",
        text,
        flags=re.S,
    )


def split_bibliography_names(raw_names: str) -> list[str]:
    return [name.strip() for name in raw_names.split(",") if name.strip()]


def find_bib_file(name: str, *, source_dir: Path, main_tex: Path) -> Path | None:
    candidates = [main_tex.parent / name, source_dir / name]
    if not name.endswith(".bib"):
        candidates.extend([main_tex.parent / f"{name}.bib", source_dir / f"{name}.bib"])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def parse_bib_entries(bib_text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    index = 0
    while index < len(bib_text):
        start = bib_text.find("@", index)
        if start == -1:
            break
        brace = bib_text.find("{", start)
        if brace == -1:
            break
        balanced = extract_balanced_brace(bib_text, brace)
        if not balanced:
            break
        body, end = balanced
        if "," not in body:
            index = end
            continue
        key, fields_raw = body.split(",", 1)
        key = key.strip()
        fields: dict[str, str] = {}
        for match in re.finditer(r"([A-Za-z]+)\s*=\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\"[^\"]*\"|[^,\n]+)", fields_raw):
            field_name = match.group(1).lower()
            value = match.group(2).strip().strip(",")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "{"}:
                value = value[1:-1]
            fields[field_name] = clean_latex_command_markup(value)
        if key:
            entries[key] = fields
        index = end
    return entries


def format_bib_reference(key: str, fields: dict[str, str]) -> str:
    author = fields.get("author", "").replace(" and ", "; ")
    title = fields.get("title", "")
    venue = fields.get("journal") or fields.get("booktitle") or fields.get("publisher") or ""
    year = fields.get("year", "")
    eprint = fields.get("eprint", "")

    pieces = [piece for piece in [author, title, venue, year] if piece]
    if eprint:
        pieces.append(f"arXiv:{eprint}")
    if not pieces:
        return f"- [{key}]"
    return f"- [{key}] " + ". ".join(pieces) + "."


def build_references_markdown(
    bibliography_names: list[str],
    *,
    source_dir: Path,
    main_tex: Path,
) -> tuple[str, int]:
    entries: dict[str, dict[str, str]] = {}
    for name in bibliography_names:
        bib_path = find_bib_file(name, source_dir=source_dir, main_tex=main_tex)
        if not bib_path:
            continue
        entries.update(parse_bib_entries(read_text_guess(bib_path)))

    if not entries:
        return "", 0

    lines = ["## References", ""]
    for key in sorted(entries):
        lines.append(format_bib_reference(key, entries[key]))
    return "\n".join(lines).strip() + "\n", len(entries)


def latex_to_markdown(
    tex: str,
    *,
    arxiv_id: str,
    main_tex: Path,
    source_dir: Path,
    output_path: Path,
) -> LatexConversionResult:
    tex = apply_simple_macros(tex, extract_simple_macros(tex))
    tex = tex.replace("{}", "")
    title = extract_command_value(tex, "title")
    author = extract_command_value(tex, "author")
    body = extract_document_body(tex)
    body = remove_layout_commands(body)
    body, figures, assets = convert_figure_environments(
        body,
        source_dir=source_dir,
        output_path=output_path,
    )
    body, tables = convert_table_environments(body)
    bibliography_names = [
        name
        for match in BIB_COMMAND_RE.finditer(body)
        for name in split_bibliography_names(match.group(1))
    ]
    references_markdown, references = build_references_markdown(
        bibliography_names,
        source_dir=source_dir,
        main_tex=main_tex,
    )
    body = convert_abstract(body)
    body = convert_display_math_envs(body)
    body = convert_inline_math(body)
    body = convert_sections(body)
    body = convert_lists(body)
    body = re.sub(r"\\maketitle", "", body)
    body = re.sub(r"\\bibliographystyle\{[^{}]*\}", "", body)
    body = BIB_COMMAND_RE.sub("", body)
    body = clean_latex_markdown(body)
    if references_markdown:
        body = body.rstrip() + "\n\n" + references_markdown.strip()

    frontmatter = [
        "---",
        f"source: arxiv:{arxiv_id}",
        f"tex_main: {main_tex.name}",
        "math_source: latex",
        "---",
        "",
    ]
    if title:
        frontmatter.extend([f"# {title}", ""])
    if author:
        frontmatter.extend([author, ""])
    frontmatter.append(body)
    markdown = "\n".join(frontmatter).strip() + "\n"
    return LatexConversionResult(
        markdown=markdown,
        figures=figures,
        tables=tables,
        assets=assets,
        references=references,
        unresolved_macros=count_unresolved_text_macros(markdown),
    )


def convert_from_arxiv_source(
    pdf_path: Path,
    *,
    output_path: Path,
    config: ConversionConfig,
) -> ConversionReport | None:
    arxiv_id = detect_arxiv_id(pdf_path)
    if not arxiv_id:
        if config.source == "require":
            raise RuntimeError(f"Could not detect an arXiv id in filename: {pdf_path.name}")
        return None

    try:
        source_path = download_arxiv_source(
            arxiv_id,
            cache_dir=config.source_cache,
            timeout=config.timeout,
        )
        with tempfile.TemporaryDirectory(prefix=f"arxiv-{arxiv_id}-") as tmp:
            source_dir = Path(tmp)
            unpack_arxiv_source(source_path, source_dir)
            main_tex = find_main_tex(source_dir)
            tex = expand_tex_inputs(main_tex)
            result = latex_to_markdown(
                tex,
                arxiv_id=arxiv_id,
                main_tex=main_tex,
                source_dir=source_dir,
                output_path=output_path,
            )
            if config.keep_raw:
                raw_path = output_path.with_name(f"{output_path.stem}.source.tex")
                write_markdown(raw_path, tex)
            write_markdown(output_path, result.markdown)
    except RuntimeError:
        if config.source == "require":
            raise
        return None

    return ConversionReport(
        pdf_path=pdf_path,
        output_path=output_path,
        source=f"arxiv:{arxiv_id}",
        math_source="latex",
        figures=result.figures,
        tables=result.tables,
        assets=result.assets,
        references=result.references,
        unresolved_macros=result.unresolved_macros,
    )


@contextlib.contextmanager
def suppress_library_output() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def extract_static_pages(pdf_path: Path) -> list[PageMarkdown]:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError("Missing dependency: run `uv sync` or `uv add pymupdf4llm`.") from exc

    with suppress_library_output():
        page_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            show_progress=False,
        )

    if not isinstance(page_chunks, list):
        text = str(page_chunks).strip()
        return [PageMarkdown(page_number=1, text=text)]

    pages: list[PageMarkdown] = []
    for index, item in enumerate(page_chunks, start=1):
        if not isinstance(item, dict):
            text = str(item).strip()
            pages.append(PageMarkdown(page_number=index, text=text))
            continue

        metadata = item.get("metadata")
        page_number = index
        if isinstance(metadata, dict) and isinstance(metadata.get("page_number"), int):
            page_number = metadata["page_number"]

        text = item.get("text")
        pages.append(PageMarkdown(page_number=page_number, text=str(text or "").strip()))

    return pages


def format_page(page: PageMarkdown) -> str:
    if page.text:
        return f"<!-- page: {page.page_number} -->\n\n{page.text.strip()}"
    return f"<!-- page: {page.page_number} -->"


def combine_pages(pages: Iterable[PageMarkdown]) -> str:
    return "\n\n".join(format_page(page) for page in pages).strip() + "\n"


def chunk_pages(pages: list[PageMarkdown], *, chunk_chars: int) -> list[MarkdownChunk]:
    chunks: list[MarkdownChunk] = []
    current_pages: list[PageMarkdown] = []
    current_size = 0

    for page in pages:
        formatted = format_page(page)
        page_size = len(formatted)
        if current_pages and current_size + page_size > chunk_chars:
            chunks.append(
                MarkdownChunk(
                    start_page=current_pages[0].page_number,
                    end_page=current_pages[-1].page_number,
                    text="\n\n".join(format_page(item) for item in current_pages),
                )
            )
            current_pages = []
            current_size = 0

        current_pages.append(page)
        current_size += page_size

    if current_pages:
        chunks.append(
            MarkdownChunk(
                start_page=current_pages[0].page_number,
                end_page=current_pages[-1].page_number,
                text="\n\n".join(format_page(item) for item in current_pages),
            )
        )

    return chunks


def looks_like_noisy_extraction(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    alnum_count = sum(char.isalnum() for char in stripped)
    if alnum_count < 250:
        return True

    replacement_count = stripped.count("\ufffd")
    if replacement_count / max(len(stripped), 1) > 0.002:
        return True

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 20:
        very_short = sum(1 for line in lines if len(line) <= 3)
        if very_short / len(lines) > 0.35:
            return True

    words = re.findall(r"[A-Za-z]{2,}", stripped)
    if words:
        one_letter_lines = sum(1 for line in lines if re.fullmatch(r"[A-Za-z]", line))
        if one_letter_lines > 25:
            return True

    return False


def clean_markdown_chunk(
    chunk: MarkdownChunk,
    *,
    api_key: str,
    model: str,
    timeout: int,
) -> str:
    prompt = (
        "You are cleaning Markdown that was extracted locally from a PDF.\n\n"
        "Rules:\n"
        "- Return only cleaned Markdown.\n"
        "- Do not summarize, omit, translate, or add new content.\n"
        "- Preserve page comments like `<!-- page: 3 -->`.\n"
        "- Fix broken line wraps, obvious OCR artifacts, heading hierarchy, lists, tables, code blocks, "
        "references, and reading order when the intent is clear.\n"
        "- Keep equations and citations as close to the source text as possible.\n\n"
        f"Pages: {chunk.start_page}-{chunk.end_page}\n\n"
        "Markdown to clean:\n\n"
        f"{chunk.text}"
    )
    response = http_json_request(
        f"{OPENAI_BASE_URL}/responses",
        api_key=api_key,
        timeout=timeout,
        payload={
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        },
    )

    cleaned = extract_output_text(response)
    if not cleaned:
        raise RuntimeError(f"The model returned no Markdown for pages {chunk.start_page}-{chunk.end_page}")
    return cleaned.strip()


def clean_chunks_with_llm(
    chunks: list[MarkdownChunk],
    *,
    clean_policy: str,
    api_key: str | None,
    model: str,
    timeout: int,
) -> tuple[str, int]:
    cleaned_chunks: list[str] = []
    llm_chunks = 0
    for index, chunk in enumerate(chunks, start=1):
        should_clean = clean_policy == "all" or (
            clean_policy == "auto" and looks_like_noisy_extraction(chunk.text)
        )
        if not should_clean:
            cleaned_chunks.append(chunk.text.strip())
            continue

        key = require_api_key(api_key)
        print(
            f"[llm-clean] chunk {index}/{len(chunks)} pages {chunk.start_page}-{chunk.end_page}",
            file=sys.stderr,
        )
        llm_chunks += 1
        cleaned_chunks.append(
            clean_markdown_chunk(chunk, api_key=key, model=model, timeout=timeout)
        )

    markdown = "\n\n".join(part for part in cleaned_chunks if part.strip()).strip() + "\n"
    return markdown, llm_chunks


def convert_static(pdf_path: Path, *, output_path: Path, keep_raw: bool) -> ConversionReport:
    pages = extract_static_pages(pdf_path)
    markdown = combine_pages(pages)
    if keep_raw:
        write_markdown(output_path.with_suffix(".raw.md"), markdown)
    write_markdown(output_path, markdown)
    return ConversionReport(
        pdf_path=pdf_path,
        output_path=output_path,
        source="pdf-static",
        math_source="pdf-extracted",
        unresolved_macros=count_unresolved_text_macros(markdown),
    )


def convert_hybrid(
    pdf_path: Path,
    *,
    output_path: Path,
    config: ConversionConfig,
) -> ConversionReport:
    pages = extract_static_pages(pdf_path)
    raw_markdown = combine_pages(pages)
    if config.keep_raw:
        write_markdown(output_path.with_suffix(".raw.md"), raw_markdown)

    if config.clean == "none":
        write_markdown(output_path, raw_markdown)
        return ConversionReport(
            pdf_path=pdf_path,
            output_path=output_path,
            source="pdf-static",
            math_source="pdf-extracted",
            unresolved_macros=count_unresolved_text_macros(raw_markdown),
        )

    chunks = chunk_pages(pages, chunk_chars=config.chunk_chars)
    markdown, llm_chunks = clean_chunks_with_llm(
        chunks,
        clean_policy=config.clean,
        api_key=config.api_key,
        model=config.model,
        timeout=config.timeout,
    )
    write_markdown(output_path, markdown)
    return ConversionReport(
        pdf_path=pdf_path,
        output_path=output_path,
        source="pdf-hybrid",
        math_source="pdf-extracted",
        unresolved_macros=count_unresolved_text_macros(markdown),
        llm_chunks=llm_chunks,
    )


def convert_pdf_with_llm(
    pdf_path: Path,
    *,
    output_path: Path,
    api_key: str,
    model: str,
    timeout: int,
) -> ConversionReport:
    if pdf_path.stat().st_size > MAX_DIRECT_INPUT_BYTES:
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        raise RuntimeError(f"PDF is {size_mb:.1f} MB; direct PDF inputs are limited to 50 MB.")

    file_id = upload_pdf(pdf_path, api_key=api_key, timeout=timeout)
    try:
        prompt = (
            "Convert the attached PDF into clean Markdown that is faithful to the original document.\n\n"
            "Rules:\n"
            "- Return only the Markdown content, with no explanations before or after it.\n"
            "- Preserve headings, subheadings, lists, tables, and references when possible.\n"
            "- Keep the page reading order.\n"
            "- Use Markdown code blocks for code snippets.\n"
            "- Do not summarize the document."
        )

        response = http_json_request(
            f"{OPENAI_BASE_URL}/responses",
            api_key=api_key,
            timeout=timeout,
            payload={
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file_id},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
            },
        )
    finally:
        delete_uploaded_file(file_id, api_key=api_key, timeout=timeout)

    markdown = extract_output_text(response)
    if not markdown:
        raise RuntimeError(f"The model response did not include Markdown text: {response}")

    write_markdown(output_path, markdown.rstrip() + "\n")
    return ConversionReport(
        pdf_path=pdf_path,
        output_path=output_path,
        source="openai-pdf",
        math_source="model",
        unresolved_macros=count_unresolved_text_macros(markdown),
        llm_chunks=1,
    )


def write_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")


def validate_pdf_path(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise RuntimeError(f"Input must be a .pdf file: {pdf_path}")


def convert_one(pdf_path: Path, *, output_path: Path, config: ConversionConfig) -> ConversionReport:
    validate_pdf_path(pdf_path)

    if config.source != "never" and (config.mode != "llm" or config.source == "require"):
        report = convert_from_arxiv_source(pdf_path, output_path=output_path, config=config)
        if report:
            print(f"[source] used arXiv LaTeX for {pdf_path.name}", file=sys.stderr)
            return report

    if config.mode == "static":
        return convert_static(pdf_path, output_path=output_path, keep_raw=config.keep_raw)

    if config.mode == "hybrid":
        return convert_hybrid(pdf_path, output_path=output_path, config=config)

    if config.mode == "llm":
        return convert_pdf_with_llm(
            pdf_path,
            output_path=output_path,
            api_key=require_api_key(config.api_key),
            model=config.model,
            timeout=config.timeout,
        )

    raise RuntimeError(f"Unsupported mode: {config.mode}")


def iter_pdfs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.pdf")
        if path.is_file() and not any(part.endswith("_assets") for part in path.parts)
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert papers to Markdown using arXiv LaTeX source, static PDF extraction, and optional LLM cleanup.",
    )
    parser.add_argument("pdf", nargs="?", type=Path, help="Path to one PDF file.")
    parser.add_argument(
        "--all",
        dest="all_dir",
        type=Path,
        help="Recursively convert all PDFs under this directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Markdown path for single-file mode. Defaults to the PDF path with .md extension.",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "hybrid", "llm"],
        default="hybrid",
        help="Conversion mode. Defaults to hybrid.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "never", "require"],
        default="auto",
        help="arXiv LaTeX source policy for preserving math. Defaults to auto.",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=DEFAULT_SOURCE_CACHE,
        help=f"Directory for cached arXiv source downloads. Defaults to {DEFAULT_SOURCE_CACHE}.",
    )
    parser.add_argument(
        "--clean",
        choices=["auto", "all", "none"],
        default="auto",
        help="Hybrid cleanup policy. Defaults to auto.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help=f"Maximum characters per LLM cleanup chunk. Defaults to {DEFAULT_CHUNK_CHARS}.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Also write the static extraction as a .raw.md file.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Path to .env file. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--model",
        help="Override OPENAI_MODEL from the environment file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="OpenAI request timeout in seconds. Defaults to 600.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite Markdown files if they already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch conversion after the first failure.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ConversionConfig:
    env_path = args.env.expanduser().resolve()
    env_file_values = load_env(env_path)
    api_key = env_value(env_file_values, "OPENAI_API_KEY")
    model = args.model or env_value(env_file_values, "OPENAI_MODEL", "gpt-5.4-mini")

    if args.chunk_chars < 2_000:
        raise RuntimeError("--chunk-chars must be at least 2000")
    if args.timeout < 30:
        raise RuntimeError("--timeout must be at least 30 seconds")
    if args.clean != "none" and args.mode == "llm":
        print("[warn] --clean is ignored in llm mode", file=sys.stderr)

    return ConversionConfig(
        mode=args.mode,
        clean=args.clean,
        source=args.source,
        source_cache=args.source_cache.expanduser().resolve(),
        chunk_chars=args.chunk_chars,
        keep_raw=args.keep_raw,
        api_key=api_key,
        model=str(model),
        timeout=args.timeout,
    )


def output_for_pdf(pdf_path: Path, output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg.expanduser().resolve()
    return pdf_path.with_suffix(".md").resolve()


def print_report(report: ConversionReport) -> None:
    print(
        "[report] "
        f"source={report.source} "
        f"math={report.math_source} "
        f"figures={report.figures} "
        f"tables={report.tables} "
        f"assets={report.assets} "
        f"references={report.references} "
        f"unresolved_macros={report.unresolved_macros} "
        f"llm_chunks={report.llm_chunks}",
        file=sys.stderr,
    )


def convert_single(args: argparse.Namespace, config: ConversionConfig) -> int:
    pdf_path = args.pdf.expanduser().resolve()
    output_path = output_for_pdf(pdf_path, args.output)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists. Use --overwrite to replace it: {output_path}", file=sys.stderr)
        return 2

    try:
        report = convert_one(pdf_path, output_path=output_path, config=config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_report(report)
    print(output_path)
    return 0


def convert_batch(args: argparse.Namespace, config: ConversionConfig) -> int:
    root = args.all_dir.expanduser().resolve()
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 2
    if args.output:
        print("--output can only be used with a single PDF", file=sys.stderr)
        return 2

    pdfs = iter_pdfs(root)
    if not pdfs:
        print(f"No PDF files found under: {root}", file=sys.stderr)
        return 2

    failures: list[tuple[Path, str]] = []
    reports: list[ConversionReport] = []
    skipped = 0
    started_at = time.perf_counter()
    for index, pdf_path in enumerate(pdfs, start=1):
        output_path = pdf_path.with_suffix(".md").resolve()
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {index}/{len(pdfs)} {output_path}")
            continue

        print(f"[convert] {index}/{len(pdfs)} {pdf_path}")
        try:
            report = convert_one(pdf_path, output_path=output_path, config=config)
        except RuntimeError as exc:
            failures.append((pdf_path, str(exc)))
            print(f"[failed] {pdf_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
            continue
        reports.append(report)
        print_report(report)
        print(f"[ok] {output_path}")

    elapsed = time.perf_counter() - started_at
    sources: dict[str, int] = {}
    for report in reports:
        sources[report.source] = sources.get(report.source, 0) + 1
    source_summary = ", ".join(f"{key}:{value}" for key, value in sorted(sources.items())) or "none"
    print(
        "[summary] "
        f"converted={len(reports)} skipped={skipped} failed={len(failures)} "
        f"sources={source_summary} "
        f"assets={sum(report.assets for report in reports)} "
        f"references={sum(report.references for report in reports)} "
        f"llm_chunks={sum(report.llm_chunks for report in reports)} "
        f"elapsed={elapsed:.2f}s",
        file=sys.stderr,
    )

    if failures:
        print(f"{len(failures)} PDF(s) failed.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if bool(args.pdf) == bool(args.all_dir):
        print("Provide either one PDF path or --all DIR.", file=sys.stderr)
        return 2

    try:
        config = build_config(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.all_dir:
        return convert_batch(args, config)
    return convert_single(args, config)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
