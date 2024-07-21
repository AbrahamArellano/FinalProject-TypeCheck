{-# LANGUAGE DataKinds #-}

module Main where

import Tensor
import Control.Monad (replicateM)

main :: IO ()
main = do
  let m1 = matrix [[1, 2], [3, 4]] :: Matrix "A" "B"
      m2 = matrix [[5, 6], [7, 8]] :: Matrix "B" "C"
      v = vector [1, 2] :: Vector "A"
      s = scalar 3.14

  putStrLn "Scalar:"
  print s

  putStrLn "\nVector:"
  print v

  putStrLn "\nMatrix m1:"
  print m1

  putStrLn "\nMatrix m2:"
  print m2

  putStrLn "\nMatrix multiplication (m1 * m2):"
  print $ matMul m1 m2

  putStrLn "\nType safety test - valid multiplication:"
  let validMul = matMul (matrix [[1, 2]] :: Matrix "X" "Y") 
                        (matrix [[3], [4]] :: Matrix "Y" "Z")
  print validMul

  putStrLn "\nType safety test - invalid multiplication (uncomment to see error):"
  -- Uncommenting the next line should result in a compile-time error
  -- let invalidMul = matMul (matrix [[1, 2]] :: Matrix "X" "Y") (matrix [[3, 4]] :: Matrix "Z" "W")

  putStrLn "\nNeural Network Demonstration:"
  nn <- initializeNN 2 3 1
  let input = vector [0.5, 0.8] :: Vector "Input"
  let output = forwardNN nn input
  putStrLn "Input:"
  print input
  putStrLn "Output:"
  print output