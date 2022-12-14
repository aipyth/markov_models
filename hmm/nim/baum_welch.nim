import std/[
  strutils,
  sequtils,
  unicode,
  strformat,
  json,
  times,
  os,
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
    data: string
    V: seq[string]
    N: int
    A: seq[seq[float64]]
    B: seq[seq[float64]]
    P: seq[float64]
    output_file: string
    min_iterations: int
    eps: float64
    do_not_re_estimate: seq[string]


let conf = to(configJson, Config)

let runes = conf.V.mapIt(toRunes(it))
var V: seq[Rune] = @[]
for rs in runes:
  for r in rs:
    V.add(r)

let data = translateObservations(toRunes(conf.data), V)

echo fmt"V: {V}"
echo fmt"Data[0..15]: {data[0..15]}"
echo fmt"Observation sequence length (T): {data.len}"

# TODO: refactor letters encoding to support decoding back
# TODO: add analysis of given matrices
# TODO: split the letters into N groups (analysis part)
# TODO: write all generated data and iterations data to files for further analysis in python

let M = conf.V.len
let N = conf.N


var hmm = newHMM(N, M)
hmm.setA(conf.A)
hmm.setB(conf.B)
hmm.setP(conf.P)
let (A_hist, B_hist, P_hist, PLog_hist, delta_hist) = hmm.fitBaumWelch(data, conf.eps, conf.min_iterations, conf.do_not_re_estimate)

var conts = %* {
  "observations": data,
  "encoding": $V,
  "model": {
    "N": hmm.N,
    "M": hmm.M,
    "A": hmm.A,
    "B": hmm.B,
    "P": hmm.P,
  },
  "evolution": {
    "a": A_hist,
    "b": B_hist,
    "p": P_hist,
    "p_log": PLog_hist,
    "delta": delta_hist,
  }
}

writeFile(conf.output_file % ["time", $getTime()], $conts)
