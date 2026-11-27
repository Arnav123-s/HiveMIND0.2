# The read of g83

Run [45723265418](https://github.com/Arnav123-s/HiveMIND/actions/runs/45723265418), recorded 2026-11-26 05:24: the body was **dropped**. the regrown body passed fewer than the body before (54 against 56)

Written by the record job (`training/reads.py`), with nobody's hand. The rules are those of `docs/DECISIONS.md`.

## The numbers

| What is measured | g83 | g81 (last record) | the body before (same seeds) |
|---|---|---|---|
| Easy kinds passed | 32 of 32 | 32 of 32 | 32 of 32 |
| Hard kinds that have lessons | 22 of 25 | 23 of 27 | 23 of 25 |
| Hard kinds never practised (held out) | 1 of 10 kinds (3 of 29 tasks of the fixed exam; pulse 0 of 7) | 1 of 10 kinds (2 of 28 tasks of the fixed exam; pulse 0 of 5) | 1 of 10 kinds (4 of 29 tasks of the fixed exam; pulse 1 of 7) |
| Right at the first try (practice) | not measured | 0.56 | - |
| Right at the first try, old kinds and K2's new kinds (practice) | not measured | old kinds 0.95 (36 of 38); new kinds (K2) 0.31 (18 of 58) | - |
| Hours for one generation | 5.7 | 5.2 | - |
| Generations recorded with nobody's hand | 1 | 1 | - |
| Talk (the talk exam, out of 10; the bar is 8) | 8.0 of 10 (route 8 of 10, fact 9 of 10, earlier 9 of 10, time 6 of 10) | 8.2 of 10 (route 9 of 10, fact 9 of 10, earlier 10 of 10, time 5 of 10) | - |

## What members solved and what reached the user

| Exam arm | Scope | Graded for both, of tasks sat | Any recorded member passed | Final delivery passed | Member passed, delivery failed | Delivery passed, no recorded member passed |
|---|---|---:|---:|---:|---:|---:|
| before | all | 64 of 64 | 58 | 56 | 2 | 0 |
| before | hard | 32 of 32 | 26 | 24 | 2 | 0 |
| after | all | 64 of 64 | 56 | 54 | 2 | 0 |
| after | hard | 32 of 32 | 24 | 22 | 2 | 0 |

Counts compare the task checker on recorded member answers and their files with the same checker on the final delivery. Only tasks with a known result on both sides enter the last four columns.

## Which final deliveries changed

Comparable deliveries: 64 matched, from 64 tasks sat before and 64 after. Matching requires the same seat, task ID, prompt, kind and exam revision.

| Gained | Lost | Passed both | Failed both |
|---:|---:|---:|---:|
| 0 | 2 | 54 | 8 |

Lost: `h20000831_00` (a markdown table turned into CSV), `h20000831_01` (two facts from two sources, combined).

## The chain, by taught hard kind

lessons right by their own check -> rows that fit in training -> loss falls on lessons kept back -> the child alone writes it at the exam -> the body (two agree) passes it.

| kind | lessons right of made | rows left out | kept-back loss before -> after | gap to untaught (points) | child alone right | parent alone right | body right | body before |
|---|---|---|---|---|---|---|---|---|
| a fact from a site and a date sum on it | none made | 15 of 100 (13 skipped for memory) | 0.000 -> 0.001 | -56 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a join over two small tables | 12 of 12 | 128 of 760 (109 skipped for memory) | 0.392 -> 0.270 | +68 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a log read and counted into a file | none made | 39 of 100 (9 skipped for memory) | 0.000 -> 0.001 | -38 | - | - | - | - |
| a module of two functions, with hidden tests | none made | 148 of 1280 (201 skipped for memory) | 0.137 -> 0.155 | +24 | 4 of 4 | 1 of 2 | 2 of 2 | 2 of 2 |
| a class with rules, with hidden tests | 12 of 12 | 54 of 760 (140 skipped for memory) | 0.012 -> 0.013 | +27 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a parser for a tiny language, with hidden tests | 12 of 12 | 50 of 640 (101 skipped for memory) | 0.007 -> 0.009 | +8 | 1 of 5 | 0 of 1 | 0 of 1 | 0 of 1 |
| the regions of a grid, with hidden tests | 13 of 13 | 149 of 820 (123 skipped for memory) | 0.002 -> 0.003 | -6 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a cache that forgets the least recently used, with hidden tests | 13 of 13 | 148 of 920 (143 skipped for memory) | 0.015 -> 0.022 | -13 | - | - | - | - |
| a state machine from a table, with hidden tests | 12 of 12 | 32 of 520 (76 skipped for memory) | 0.001 -> 0.003 | -58 | 0 of 4 | 0 of 1 | 0 of 1 | 0 of 1 |
| an order of tasks by their dependencies, with hidden tests | 12 of 12 | 86 of 960 (192 skipped for memory) | 0.003 -> 0.003 | +18 | 6 of 6 | 3 of 3 | 3 of 3 | 3 of 3 |
| a number with given properties | none made | 0 of 140 (3 skipped for memory) | not measured | not measured | - | - | - | - |
| a fact read and a date sum on it | none made | 10 of 120 (9 skipped for memory) | not measured | not measured | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| the percent change between two columns, rounded by a rule | none made | 31 of 280 (65 skipped for memory) | 0.001 -> 0.004 | -245 | 4 of 4 | 2 of 2 | 2 of 2 | 2 of 2 |
| a table cut to one row a key, the latest kept | none made | 42 of 400 (70 skipped for memory) | 0.002 -> 0.004 | -87 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| the blanks of a table filled by a rule | none made | 45 of 320 (55 skipped for memory) | 0.001 -> 0.002 | -187 | 4 of 4 | 2 of 2 | 2 of 2 | 2 of 2 |
| a table grouped, each group's total and mean saved | none made | 12 of 80 (11 skipped for memory) | 0.001 -> 0.002 | -86 | - | - | - | - |
| JSON lines read and counted by a nested field | 12 of 12 | 121 of 780 (49 skipped for memory) | 0.000 -> 0.000 | -159 | - | - | - | - |
| a long table turned wide | none made | 14 of 200 (42 skipped for memory) | 0.001 -> 0.003 | -324 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| several facts fetched and laid out as a file | none made | 6 of 80 (13 skipped for memory) | 0.001 -> 0.001 | -50 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a server log's matching lines counted by class | 12 of 12 | 672 of 1152 (41 skipped for memory) | 0.000 -> 0.001 | -326 | - | - | - | - |
| emails and phones found in a messy message | none made | 35 of 480 (78 skipped for memory) | 0.023 -> 0.020 | +53 | 2 of 2 | 1 of 1 | 1 of 1 | 1 of 1 |
| a markdown table turned into CSV | 12 of 12 | 69 of 740 (93 skipped for memory) | 0.001 -> 0.001 | +28 | 2 of 6 | 2 of 3 | 2 of 3 | 3 of 3 |
| names turned and sorted | 15 of 15 | 8 of 700 (75 skipped for memory) | 0.000 -> 0.001 | -36 | - | - | - | - |
| a text split into sentences | 12 of 12 | 21 of 640 (67 skipped for memory) | 0.034 -> 0.038 | +26 | - | - | - | - |
| the most frequent words of a text | 12 of 12 | 20 of 1040 (115 skipped for memory) | 0.141 -> 0.093 | +71 | - | - | - | - |
| walk a site over several pages | none made | 2 of 20 (6 skipped for memory) | 0.000 -> 0.005 | -1576 | - | - | - | - |
| a fact found on the open web | none made | 15 of 220 (14 skipped for memory) | 0.002 -> 0.005 | -161 | 4 of 4 | 2 of 2 | 2 of 2 | 2 of 2 |

- rows left out: by kind, this generation's rows too long for a slice, of those the slices were given; in brackets those the memory guard skipped every time they came up. Rows not reached before the time was up are not left out
- gap: the kind's fall beside the mean fall of the kinds none of whose lessons were trained on (operate a website: -37 points)
- child alone: the answers of this generation's cells (4) in the pulse; parent alone: those of their parents (g81code2, g81data, g81sources, g81words2) in every arm. Two members sit a task, so these are few

How much the hive taught itself: of this generation's new rows, 372 from the hive's own checked practice and dreams (4 of them fix rows), 387 from the hand-written lesson makers (training/lessons.py, training/kinds/): the hive's own are 49% of those two. Rows trained on: not measured by source (each slice's record says the rows it trained by kind only, rows_by_kind; a field rows_by_source {source: trained} would measure it).

![The chain by kind](../progress/chain.svg)

## Hours

| step | minutes |
|---|---|
| practice | 44 |
| dream | 51 |
| gather | 4 |
| plan | 0 |
| train | 177 |
| join | 2 |
| gate | 105 |
| record | 3 |

From the first job to the last: **5.71 hours** (the bar is 3). Each step runs from its first job's start to its last job's end; practice and dream run side by side.

Train slices lost: 0 of 20. Pulse seats whose server died: 0.

Colab: not used (notebook not seen)

## Next piece

next piece: dose: one change a generation: raw reading and judged pairs off; slices weighed by steps; the one-line trial (rule 3: kept-back gap under 15 points, for 17 of 27 taught hard kinds: a fact from a site and a date sum on it, a log read and counted into a file, a parser for a tiny language, with hidden tests, the regions of a grid, with hidden tests, a cache that forgets the least recently used, with hidden tests, a state machine from a table, with hidden tests, the percent change between two columns, rounded by a rule, a table cut to one row a key, the latest kept, the blanks of a table filled by a rule, a table grouped, each group's total and mean saved, JSON lines read and counted by a nested field, a long table turned wide, several facts fetched and laid out as a file, a server log's matching lines counted by class, names turned and sorted, walk a site over several pages, a fact found on the open web)

Rules that hold but whose piece is done on the board (docs/PLAN.md), so not named: rule 2 (T4); rule 4 (R1); rule 6 (K1).

| # | the chain shows | holds for |
|---|---|---|
| 1 | lessons right under 8 of 12 | 0 |
| 2 | more than 1 row in 5 of the kind left out | 2: a log read and counted into a file, a server log's matching lines counted by class (piece done) |
| 3 | kept-back gap under 15 points | 17: a fact from a site and a date sum on it, a log read and counted into a file, a parser for a tiny language, with hidden tests, the regions of a grid, with hidden tests, a cache that forgets the least recently used, with hidden tests, a state machine from a table, with hidden tests, the percent change between two columns, rounded by a rule, a table cut to one row a key, the latest kept, the blanks of a table filled by a rule, a table grouped, each group's total and mean saved, JSON lines read and counted by a nested field, a long table turned wide, several facts fetched and laid out as a file, a server log's matching lines counted by class, names turned and sorted, walk a site over several pages, a fact found on the open web |
| 4 | gap 15 or more, the child alone right more often than its parent, the body not | 1: a module of two functions, with hidden tests (piece done) |
| 5 | gap 15 or more, the child alone not right | 0 |
| 6 | taught hard kinds 12 of 23, two generations running | 27 (piece done) |
