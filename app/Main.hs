{-# LANGUAGE DataKinds #-}

module Main where

import Tensor

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

  putStrLn "\nAttempting invalid multiplication (uncomment to see error):"
  --print $ matMul m1 m1  -- This should cause a type error