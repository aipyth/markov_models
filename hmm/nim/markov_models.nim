import sequtils
import strformat
import strutils
import terminal
import nimpy
import times
import math
import os


import hmm_utils

type
  HMM* = object
    N*: int
    M*: int
    A*: seq[seq[float64]]
    B*: seq[seq[float64]]
    P*: seq[float64]

proc newHMM*(N: int, M: int): HMM = 
  HMM(
    N: N,
    M: M,
    A: newSeqWith(N, newSeq[float64](N)),
    B: newSeqWith(N, newSeq[float64](M)),
    P: newSeq[float64](N),
  )

proc setA*(self: var HMM, A: seq[seq[float64]]) =
  self.A = A
proc setB*(self: var HMM, B: seq[seq[float64]]) =
  self.B = B
proc setP*(self: var HMM, P: seq[float64]) =
  self.P = P

proc fitBaumWelch*(self: var HMM, O: seq[int], eps: float64 = 1e-4, minIters: int = 30, do_not_re_estimate: seq[string] = @[]): tuple[
  A_hist: seq[seq[seq[float64]]],
  B_hist: seq[seq[seq[float64]]],
  P_hist: seq[seq[float64]],
  PLog_hist: seq[float64],
  delta_hist: seq[float64]
  ] =
  var A: seq[seq[float64]]
  var B: seq[seq[float64]]
  var P: seq[float64]

  A = self.A
  B = self.B
  P = self.P

  var iters = 0
  var
    A_hist = newSeq[seq[seq[float64]]](0)
    B_hist = newSeq[seq[seq[float64]]](0)
    P_hist = newSeq[seq[float64]](0)
    PLog_hist = newSeq[float64](0)
    delta_hist = newSeq[float64](0)

  var
    alpha: seq[seq[float64]]
    beta: seq[seq[float64]]
    C: seq[float64]
    gamma: seq[seq[float64]]
    digamma: seq[seq[seq[float64]]]
    pLog: float64
    oldPLog: float64
    delta: float64

  var start = times.cpuTime()
  var itsps = newSeq[float]()
  var notifiedApproxItersPerSec = false
  while true:
    (alpha, C) = hmm_utils.alphaPass(A, B, P, O)
    beta = hmm_utils.betaPass(A, B, C, O)
    (gamma, digamma) = hmm_utils.gammaCompute(A, B, O, alpha, beta)
    (A, B, P) = hmm_utils.reestimate(A, B, O, alpha, beta, gamma, digamma, do_not_re_estimate)
    pLog = hmm_utils.computePLog(C, O.len)
    if "P" in do_not_re_estimate:
      P = self.P

    # echo fmt"{iters=}"
    # echo fmt"{A=}"
    # echo fmt"{B=}"
    # echo fmt"{P=}"
    # sleep(5000)

    iters += 1
    delta = abs(pLog - oldPLog)
    # if iters mod 10 == 1:
    #   stdout.write("$1: $2" % [$iters, $delta])
    #   eraseLine()
    itsps.add(float(iters) / (times.cpuTime() - start))
    if iters > minIters and not notifiedApproxItersPerSec:
      echo fmt"Passed {minIters} iterations. Expected {float(iters) / (times.cpuTime() - start)} iterations per second."
      notifiedApproxItersPerSec = true
    if iters > minIters and delta < eps:
    # if iters > minIters:
      let its = sum(itsps) / float(itsps.len)
      echo "Average iterations per second: ", its
      self.setA(A)
      self.setB(B)
      self.setP(P)
      return (
        A_hist: A_hist,
        B_hist: B_hist,
        P_hist: P_hist,
        PLog_hist: PLog_hist,
        delta_hist: delta_hist
        )

    oldPLog = pLog
    

    A_hist.add(A)
    B_hist.add(B)
    P_hist.add(P)
    PLog_hist.add(pLog)
    delta_hist.add(delta)
  stdout.resetAttributes()

