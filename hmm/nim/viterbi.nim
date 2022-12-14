import std/[
  strutils,
  sequtils,
  unicode,
  strformat,
  json,
  times,
  os,
  math,
  ]

import hmm_utils
import markov_models

let argv = commandLineParams()
if argv.len != 1:
  echo "Invalid number of parameters"
  quit(0)

let confFilename = argv[0]

var configContents: string
try:
  configContents = readFile(confFilename)
except IOError:
  echo fmt"Error opening {confFilename}"
  quit(1)
let configJson = parseJson(configContents)

type
  Config = object
    data: seq[int]
    N: int
    # M: int
    A: seq[seq[float64]]
    B: seq[seq[float64]]
    P: seq[float64]
    output_file: string
    
    # min_iterations: int
    # eps: float64


let conf = to(configJson, Config)

# let runes = conf.V.mapIt(toRunes(it))
# var V: seq[Rune] = @[]
# for rs in runes:
#   for r in rs:
#     V.add(r)

let data = conf.data
# let data = translateObservations(toRunes(conf.data), V)

# echo fmt"V: {V}"
# echo fmt"Data[0..15]: {data[0..15]}"
echo fmt"Observation sequence length (T): {data.len}"


let N = conf.N
let T = data.len
let A = conf.A
let B = conf.B
let P = conf.P

# ALTERNATIVE VITERBI ALGORITHM

var delta_h = newSeqWith(T, newSeq[float64](N))
var psi = newSeqWith(T, newSeq[int](N))

# 1. Preprocessing (for ready up log matrices)
var p_log = newSeq[float64](N)
for i in 0 ..< N:
  p_log[i] = if P[i] != 0: log(P[i], E) else: MinFloatNormal

var a_log = newSeqWith(N, newSeq[float64](N))
for i in 0 ..< N:
  for j in 0 ..< N:
    a_log[i][j] = if A[i][j] != 0: log(A[i][j], E) else: MinFloatNormal

# 2. Initialization
for i in 0 ..< N:
  delta_h[0][i] = p_log[i] + B[i][data[0]]
  psi[0][i] = 0

# 3. Induction
var delta_tp: float64
for t in 1 ..< T:
  for i in 0 ..< N:
    # find maximum
    delta_h[t][i] = delta_h[t-1][0] + log(A[0][i], E)
    psi[t][i] = 0
    for j in 1 ..< N:
      delta_tp = delta_h[t-1][j] + a_log[i][j]
      if delta_tp > delta_h[t][i]:
        delta_h[t][i] = delta_tp
        psi[t][i] = j
    # and add missing log(B_j(O_t))
    delta_h[t][i] += log(B[i][data[t]], E)
  
# 4. Termination
var P_h: float64 = delta_h[T-1][0]
var p: float64 = delta_h[T-1][0]
var q_T: int = 0
for i in 1 ..< N:
  p = delta_h[T-1][i]
  if p > P_h:
    P_h = p
    q_T = i

# 5. Backtracking
var q = newSeq[int](T)
q[T-1] = q_T
for t in countdown(T-2, 0):
  q[t] = psi[t+1][q[t+1]]



var conts = %* {
  "delta": delta_h,
  "q": q,
  "P": P_h,
}

writeFile(conf.output_file % ["time", $getTime()], $conts)
