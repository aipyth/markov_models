import nimpy
import std/[
  sequtils,
  unicode,
  math,
]


proc translateObservations*(obs: seq[Rune], order: seq[Rune]): seq[int] =
  var output: seq[int] = @[]
  for i in obs:
    output.add(order.find(i))
  return output


proc alphaPass*(A: seq[seq[float]], B: seq[seq[float64]], P: seq[float64], O: seq[int]): tuple[alpha: seq[seq[float64]], c: seq[float64]] {.exportpy.} =
  let
    N = A.len
    T = O.len
  # 1. Initialization
  var c = newSeq[float64](O.len)
  var alpha = newSeqWith(O.len, newSeq[float64](N))
  for i in 0..<N:
    alpha[0][i] = P[i] * B[i][O[0]]
    c[0] = c[0] + alpha[0][i]
  c[0] = 1 / c[0]
  for i in 0..<N:
    alpha[0][i] = alpha[0][i] * c[0]
  # 2. Induction
  for t in 1..<T:
    for i in 0..<N:
      alpha[t][i] = 0
      for j in 0..<N:
        alpha[t][i] = alpha[t][i] + alpha[t-1][j] * A[j][i]
      alpha[t][i] = alpha[t][i] * B[i][O[t]]
      c[t] = c[t] + alpha[t][i]
    c[t] = 1 / c[t]
    for i in 0..<N:
      alpha[t][i] = alpha[t][i] * c[t]
  return (alpha: alpha, c: c)

proc betaPass*(A: seq[seq[float]], B: seq[seq[float64]], C: seq[float64], O: seq[int]): seq[seq[float64]] =
  let
    N = A.len
    T = O.len
  # 1. Initialization
  var beta = newSeqWith(T, newSeq[float64](N))
  for i in 0..<N:
    beta[T-1][i] = C[T-1]
  # 2. Induction
  for t in countdown(T-2, 0):
    for i in 0..<N:
      beta[t][i] = 0
      for j in 0..<N:
        beta[t][i] = beta[t][i] +
          A[i][j] * B[j][O[t+1]] * beta[t+1][j]
      beta[t][i] = beta[t][i] * C[t]
  return beta

proc gammaCompute*(A: seq[seq[float]], B: seq[seq[float64]], O: seq[int], alpha: seq[seq[float64]], beta: seq[seq[float64]]): tuple[gamma: seq[seq[float64]], digamma: seq[seq[seq[float64]]]] =
  let
    N = A.len
    T = O.len
  var
    gamma = newSeqWith(T, newSeq[float64](N))
    digamma = newSeqWith(T, newSeqWith(N, newSeq[float64](N)))
    dn: float64
  for t in 0 .. T-2:
    dn = 0
    for i in 0 ..< N:
      for j in 0 ..< N:
        dn += alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
    for i in 0 ..< N:
      gamma[t][i] = 0
      for j in 0 ..< N:
        digamma[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]) / dn
        gamma[t][i] = gamma[t][i] + digamma[t][i][j]
  
  dn = 0
  for i in 0 ..< N:
    dn += alpha[T-1][i]
  for i in 0 ..< N:
    gamma[T-1][i] = alpha[T-1][i] / dn

  return (gamma: gamma, digamma: digamma)

proc reestimate*(A: seq[seq[float]], B: seq[seq[float64]], O: seq[int], alpha: seq[seq[float64]], beta: seq[seq[float64]], gamma: seq[seq[float64]], digamma: seq[seq[seq[float64]]], do_not_re_estimate: seq[string]): tuple[A: seq[seq[float64]], B: seq[seq[float64]], P: seq[float64]] =
  let
    N = A.len
    M = B[0].len
    T = O.len
  var
    # Ar = newSeqWith(N, newSeq[float64](N))
    # Br = newSeqWith(N, newSeq[float64](M))
    Ar = deepCopy(A)
    Br = deepCopy(B)
    Pr = newSeq[float64](N)
  if not ("PI" in do_not_re_estimate):
    # Reestimate PI
    for i in 0 ..< N:
      Pr[i] = gamma[0][i]
  
  # Reestimate A
  var
    nm: float64
    dn: float64
  if not ("A" in do_not_re_estimate):
    for i in 0 ..< N:
      for j in 0 ..< N:
        nm = 0
        dn = 0
        for t in 0 .. T-2:
          nm += digamma[t][i][j]
          dn += gamma[t][i]
        
        Ar[i][j] = nm / dn

  if not ("B" in do_not_re_estimate):
    # Reestimate B
    for i in 0 ..< N:
      for j in 0 ..< M:
        nm = 0
        dn = 0
        for t in 0 ..< T:
          if O[t] == j:
            nm += gamma[t][i]
          dn += gamma[t][i]
        Br[i][j] = nm / dn
  return (A: Ar, B: Br, P: Pr)

proc computePLog*(C: seq[float64], T: int): float64 =
  var lp: float64 = 0
  for i in 0 ..< T:
    lp += math.log(C[i], math.E)
  return -lp
