# Dicionario de variaveis - CirCor DigiScope Phonocardiogram Dataset v1.0.3

Fonte principal: https://physionet.org/content/circor-heart-sound/1.0.3/

Este documento descreve as variaveis e campos presentes no dataset local `circor-heart-sound-1.0.3`. A terminologia segue a documentacao oficial do PhysioNet e os nomes de colunas encontrados em `training_data.csv`.

## Estrutura dos arquivos

O dataset contem, por sujeito:

- Arquivos `.wav`: gravacoes de fonocardiograma, uma por local de ausculta e, em alguns casos, mais de uma por local.
- Arquivos `.hea`: cabecalhos em formato WFDB que descrevem o respectivo `.wav`.
- Arquivos `.tsv`: anotacoes de segmentacao para o respectivo `.wav`.
- Arquivos `.txt`: descricao do sujeito, dados demograficos, dados clinicos e descricao de sopros.

Arquivos globais:

- `training_data.csv`: tabela com uma linha por sujeito e as mesmas tags dos arquivos `.txt`, em formato colunar.
- `RECORDS`: lista de registros de audio.
- `SHA256SUMS.txt`: checksums oficiais dos arquivos.
- `LICENSE.txt`: licenca do dataset.

## Convencoes de nomes

Formato geral:

- `ABCDE.txt`: arquivo de descricao do sujeito, em que `ABCDE` e o identificador numerico do sujeito.
- `ABCDE_XY.wav`: audio de um local de ausculta.
- `ABCDE_XY.hea`: cabecalho WFDB do audio correspondente.
- `ABCDE_XY.tsv`: segmentacao do audio correspondente.
- `ABCDE_XY_n.*`: gravacao repetida para o mesmo local, quando existe mais de uma gravacao; `n` e um indice inteiro.

Codigos de local de ausculta:

| Codigo | Significado |
|---|---|
| `AV` | Ponto da valva aortica |
| `PV` | Ponto da valva pulmonar |
| `TV` | Ponto da valva tricuspide |
| `MV` | Ponto da valva mitral |
| `Phc` | Outro local de ausculta |

As gravacoes de diferentes locais do mesmo sujeito foram obtidas sequencialmente na mesma sessao. Elas podem ter duracoes diferentes e nao devem ser assumidas como sincronizadas no tempo.

## Variaveis de `training_data.csv`

Cada linha representa um sujeito. A coluna `Recording locations:` tem dois-pontos no nome no arquivo CSV original.

| Variavel | Tipo | Descricao | Valores ou formato esperado |
|---|---:|---|---|
| `Patient ID` | inteiro/string | Identificador numerico do sujeito. Corresponde ao prefixo dos arquivos, por exemplo `13918` em `13918.txt`. | ID do sujeito |
| `Recording locations:` | string | Locais de ausculta disponiveis para o sujeito. | Combinacao de `AV`, `PV`, `TV`, `MV`, `Phc` separada por `+`, por exemplo `AV+PV+TV+MV` |
| `Age` | string | Categoria etaria do sujeito. | `Neonate`, `Infant`, `Child`, `Adolescent`, `Young adult`; pode aparecer como `nan` quando ausente |
| `Sex` | string | Sexo reportado no momento da aquisicao. | `Female`, `Male`; pode aparecer como `nan` quando ausente |
| `Height` | numero | Altura do sujeito em centimetros. | Valor maior que 0; pode aparecer como `nan` |
| `Weight` | numero | Peso do sujeito em quilogramas. | Valor maior que 0; pode aparecer como `nan` |
| `Pregnancy status` | boolean/string | Indica se a pessoa reportou gravidez no momento do exame. | `True`, `False`; pode aparecer como `nan` |
| `Murmur` | string | Indica se o anotador identificou sopro, nao identificou sopro ou nao conseguiu determinar. | `Present`, `Absent`, `Unknown` |
| `Murmur locations` | string | Locais de ausculta em que pelo menos um sopro foi observado. | Combinacao de `PV`, `TV`, `AV`, `MV`, `Phc` separada por `+`; `nan` quando nao aplicavel |
| `Most audible location` | string | Local de ausculta em que o sopro foi percebido com maior intensidade. | `PV`, `TV`, `AV`, `MV`, `Phc`; `nan` quando nao aplicavel |
| `Systolic murmur timing` | string | Posicao temporal do sopro dentro do periodo sistolico. | `Early-systolic`, `Mid-systolic`, `Late-systolic`, `Holosystolic`; `nan` quando nao aplicavel |
| `Systolic murmur shape` | string | Forma da intensidade do sopro sistolico ao longo do tempo. | `Crescendo`, `Decrescendo`, `Diamond`, `Plateau`; `nan` quando nao aplicavel |
| `Systolic murmur grading` | string | Graduacao do sopro sistolico segundo escala de Levine adaptada no dataset. | `I/VI`, `II/VI`, `III/VI`; `nan` quando nao aplicavel |
| `Systolic murmur pitch` | string | Pitch do sopro sistolico. | `Low`, `Medium`, `High`; `nan` quando nao aplicavel |
| `Systolic murmur quality` | string | Qualidade timbrica do sopro sistolico. | `Blowing`, `Harsh`, `Musical`; `nan` quando nao aplicavel |
| `Diastolic murmur timing` | string | Posicao temporal do sopro dentro do periodo diastolico. | `Early-diastolic`, `Mid-diastolic`, `Holodiastolic`; `nan` quando nao aplicavel |
| `Diastolic murmur shape` | string | Forma da intensidade do sopro diastolico ao longo do tempo. | `Decrescendo`, `Plateau`; `nan` quando nao aplicavel |
| `Diastolic murmur grading` | string | Graduacao do sopro diastolico em escala de I a IV, conforme a documentacao. | `I/IV`, `II/IV`, `III/IV`; `nan` quando nao aplicavel |
| `Diastolic murmur pitch` | string | Pitch do sopro diastolico. | `Low`, `Medium`, `High`; `nan` quando nao aplicavel |
| `Diastolic murmur quality` | string | Qualidade timbrica do sopro diastolico. | `Blowing`, `Harsh`; `nan` quando nao aplicavel |
| `Outcome` | string | Diagnostico global do cardiologista pediatrico sobre a condicao cardiaca do paciente. | `Normal`, `Abnormal` |
| `Campaign` | string | Campanha de triagem em que o sujeito participou. | `CC2014`, `CC2015` |
| `Additional ID` | inteiro/string | Segundo identificador de sujeitos que participaram das duas campanhas com IDs diferentes. | ID adicional do sujeito; `nan` quando nao aplicavel |

## Detalhes das categorias clinicas

### `Age`

Categorias etarias usadas conforme a terminologia pediatrica NICHD citada na documentacao:

| Valor | Descricao |
|---|---|
| `Neonate` | Nascimento ate 27 dias |
| `Infant` | 28 dias ate 1 ano |
| `Child` | 1 a 11 anos |
| `Adolescent` | 12 a 18 anos |
| `Young adult` | 19 a 21 anos |

### `Murmur`

| Valor | Descricao |
|---|---|
| `Present` | Sopros foram detectados em pelo menos uma gravacao cardiaca. |
| `Absent` | Sopros nao foram detectados em nenhuma gravacao cardiaca. |
| `Unknown` | A presenca ou ausencia de sopro nao ficou clara para o anotador. |

### Variaveis de sopro sistolico

| Variavel | Valores | Interpretacao |
|---|---|---|
| `Systolic murmur timing` | `Early-systolic`, `Mid-systolic`, `Late-systolic`, `Holosystolic` | Inicio, meio, fim ou todo o periodo sistolico. |
| `Systolic murmur shape` | `Crescendo`, `Decrescendo`, `Diamond`, `Plateau` | Intensidade aumentando, diminuindo, aumentando e depois diminuindo, ou aproximadamente constante. |
| `Systolic murmur pitch` | `Low`, `Medium`, `High` | Frequencia percebida do sopro. Em geral, pitches maiores podem refletir maior gradiente de pressao. |
| `Systolic murmur grading` | `I/VI`, `II/VI`, `III/VI` | Grau I: pouco audivel; grau II: suave, mas facilmente audivel; grau III: moderadamente alto ou alto. No dataset, `III/VI` agrupa graus III/VI e superiores. |
| `Systolic murmur quality` | `Blowing`, `Harsh`, `Musical` | Qualidade sonora percebida do sopro. |

### Variaveis de sopro diastolico

| Variavel | Valores | Interpretacao |
|---|---|---|
| `Diastolic murmur timing` | `Early-diastolic`, `Mid-diastolic`, `Holodiastolic` | Inicio, meio ou todo o periodo diastolico. |
| `Diastolic murmur shape` | `Decrescendo`, `Plateau` | Intensidade diminuindo ou aproximadamente constante. |
| `Diastolic murmur pitch` | `Low`, `Medium`, `High` | Frequencia percebida do sopro. |
| `Diastolic murmur grading` | `I/IV`, `II/IV`, `III/IV` | Escala de I a IV para sopros diastolicos. No dataset, `III/IV` agrupa graus III/IV e IV/IV. |
| `Diastolic murmur quality` | `Blowing`, `Harsh` | Qualidade sonora percebida do sopro. |

## Arquivos de descricao do sujeito (`training_data/*.txt`)

Cada arquivo `ABCDE.txt` descreve um sujeito.

### Primeira linha

Exemplo:

```text
13918 4 4000
```

| Campo | Tipo | Descricao |
|---|---:|---|
| `subject_id` | inteiro/string | Identificador do sujeito. |
| `number_of_recordings` | inteiro | Numero de gravacoes associadas ao sujeito. |
| `sampling_frequency_hz` | numero | Frequencia de amostragem das gravacoes, em Hz. No dataset local, os exemplos observados usam `4000`. |

### Linhas de gravacao

Exemplo:

```text
AV 13918_AV.hea 13918_AV.wav 13918_AV.tsv
```

| Campo | Tipo | Descricao |
|---|---:|---|
| `recording_location` | string | Local de ausculta da gravacao: `AV`, `PV`, `TV`, `MV` ou `Phc`. |
| `header_file` | string | Nome do arquivo `.hea` correspondente. |
| `wav_file` | string | Nome do arquivo `.wav` correspondente. |
| `segmentation_file` | string | Nome do arquivo `.tsv` com anotacoes de segmentacao. |

### Tags iniciadas por `#`

As demais linhas comecam com `#` e usam o formato:

```text
#Nome da variavel: valor
```

Essas tags correspondem as variaveis descritas na secao de `training_data.csv`, exceto `Patient ID` e `Recording locations:`, que sao representados pela primeira linha e pelas linhas de gravacao do arquivo `.txt`.

## Arquivos de segmentacao (`training_data/*.tsv`)

Os arquivos `.tsv` nao possuem cabecalho. Cada linha representa um intervalo temporal anotado no sinal.

| Coluna | Nome recomendado | Tipo | Unidade | Descricao |
|---:|---|---:|---|---|
| 1 | `start_time` | numero | segundos | Instante inicial do intervalo anotado. |
| 2 | `end_time` | numero | segundos | Instante final do intervalo anotado. |
| 3 | `label` | inteiro | sem unidade | Codigo da fase cardiaca ou de trecho nao anotado. |

Codigos de `label`:

| Codigo | Significado |
|---:|---|
| `0` | Segmento nao anotado do sinal |
| `1` | Onda S1 |
| `2` | Periodo sistolico |
| `3` | Onda S2 |
| `4` | Periodo diastolico |

## Arquivos de cabecalho (`training_data/*.hea`)

Os arquivos `.hea` seguem o formato WFDB. Nos arquivos locais, a estrutura observada e:

```text
13918_AV 1 4000 41152
13918_AV.wav 16+44 1 16 0 0 0 0 AV
```

### Linha de registro

| Campo | Tipo | Descricao |
|---|---:|---|
| `record_name` | string | Nome base do registro, geralmente igual ao nome do arquivo sem extensao. |
| `number_of_signals` | inteiro | Numero de sinais no registro. Para estes arquivos de audio, normalmente `1`. |
| `sampling_frequency_hz` | numero | Frequencia de amostragem em Hz. |
| `number_of_samples` | inteiro | Numero de amostras no sinal. |

### Linha do sinal

| Campo | Tipo | Descricao |
|---|---:|---|
| `file_name` | string | Arquivo de sinal associado, geralmente `.wav`. |
| `format` | string | Formato WFDB do sinal. No exemplo, `16+44` indica amostras de 16 bits com deslocamento de cabecalho WAV. |
| `gain` | numero | Ganho ADC informado no cabecalho WFDB. |
| `adc_resolution_bits` | inteiro | Resolucao ADC em bits. |
| `adc_zero` | inteiro | Valor ADC correspondente ao zero fisico. |
| `initial_value` | inteiro | Valor inicial do sinal. |
| `checksum` | inteiro | Checksum do sinal conforme WFDB. |
| `block_size` | inteiro | Tamanho de bloco conforme WFDB. |
| `description` | string | Descricao do sinal; neste dataset, costuma indicar o local de ausculta (`AV`, `PV`, `TV`, `MV` ou `Phc`). |

## Arquivos de audio (`training_data/*.wav`)

Cada `.wav` contem o sinal de fonocardiograma de um local de ausculta. O arquivo nao possui variaveis tabulares, mas seus metadados principais sao descritos pelo `.hea` correspondente:

- local de ausculta;
- frequencia de amostragem;
- numero de amostras;
- formato/resolucao do sinal.

## Valores ausentes

O valor `nan` e usado como marcador textual de ausencia ou nao aplicabilidade em varias variaveis, especialmente nas variaveis de caracterizacao de sopro quando nenhum sopro foi detectado ou quando a caracterizacao nao se aplica.

## Observacoes de uso

- `Murmur` e uma anotacao baseada nas gravacoes digitais de ausculta.
- `Outcome` e o diagnostico global do cardiologista pediatrico com base no processo clinico completo; ele nao e necessariamente equivalente a presenca de sopro digital.
- Nem todos os sujeitos possuem gravacoes nos quatro locais principais.
- Quando ha mais de uma gravacao para o mesmo local, o indice no nome do arquivo diferencia cada tomada, por exemplo `ABCDE_AV_1.wav` e `ABCDE_AV_2.wav`.
