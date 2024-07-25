{-# LANGUAGE DataKinds #-}
module Main where

import Tensor
import Tests
import Control.Monad (forM_)
import System.IO (hFlush, stdout)

main :: IO ()
main = do
    putStrLn "===== Running Tests ====="
    testResults <- runTests
    putStrLn $ replicate 30 '-'
    putStrLn $ "Test Summary: " ++ show testResults
    putStrLn $ replicate 30 '-'
    
    putStrLn "\n===== Neural Network Demonstration ====="
    nn <- initializeNN 2 3 1
    let inputs = [vector [0.5, 0.8], vector [0.2, 0.9], vector [0.1, 0.3]] :: [Vector "Input"]
    
    putStrLn "Neural Network Structure:"
    print nn
    
    putStrLn "\nProcessing inputs through the neural network:"
    forM_ (zip [1..] inputs) $ \(i, input) -> do
        let output = forwardNN nn input
        putStrLn $ replicate 20 '-'
        putStrLn $ "Input " ++ show i ++ ":"
        print (input :: Vector "Input")
        putStrLn "Output:"
        print (output :: Vector "Output")
        hFlush stdout

    putStrLn $ replicate 20 '-'
    putStrLn "Note: The neural network weights are randomly initialized, so outputs will vary."
    
    putStrLn "\n===== Type Safety Demonstration ====="
    putStrLn "The following examples demonstrate type safety. Uncomment them one at a time to see the compilation errors."

    -- Example 1: Invalid matrix multiplication (dimension mismatch)
    -- Uncomment the next line to see the error:
    -- let invalidMul = matMul (matrix [[1, 2]] :: Matrix "X" "Y") (vector [3, 4] :: Vector "Z")

    -- Example 2: Mismatched neural network types
    -- Uncomment the next line to see the error:
    let invalidNN = initializeNN 2 3 1 :: FeedForwardNN "Input" "Hidden" "Output"

    -- Example 3: Incorrect input vector dimensionality
    -- Uncomment the next two lines to see the error:
    -- let invalidInput = vector [1.0, 2.0, 3.0] :: Vector "InvalidInput"
    -- let invalidOutput = forwardNN nn invalidInput

    -- Example 4: Adding vectors of different dimensions
    -- Uncomment the next three lines to see the error:
    -- let v1 = vector [1, 2] :: Vector "A"
    -- let v2 = vector [3, 4, 5] :: Vector "B"
    -- let invalidAdd = addVectors v1 v2

    -- Example 5: Using a matrix where a vector is expected
    -- Uncomment the next two lines to see the error:
    -- let m = matrix [[1, 2], [3, 4]] :: Matrix "X" "Y"
    -- let invalidForward = forwardNN nn m

    putStrLn "These examples demonstrate how the type system prevents common errors in tensor operations."