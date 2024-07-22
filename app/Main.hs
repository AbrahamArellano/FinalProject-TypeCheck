{-# LANGUAGE DataKinds #-}

module Main where

import Tensor
import Tests
import Control.Monad (forM_)

main :: IO ()
main = do
    -- Execute the test suite
    putStrLn "Running tests..."
    _ <- runTests
    putStrLn "\nNeural Network Demonstration:"

    -- Initialize a simple feed-forward neural network
    -- with 2 input neurons, 3 hidden neurons, and 1 output neuron
    nn <- initializeNN 2 3 1

    -- Prepare a set of input vectors for demonstration
    let inputs = [vector [0.5, 0.8], vector [0.2, 0.9], vector [0.1, 0.3]] :: [Vector "Input"]
    
    putStrLn "Processing inputs through the neural network:"

    -- Iterate through the inputs, passing each through the neural network
    -- and displaying the results
    forM_ (zip [(1::Int)..] inputs) $ \(i, input) -> do
        let output = forwardNN nn input
        putStrLn $ "Input " ++ show i ++ ":"
        print (input :: Vector "Input")
        putStrLn $ "Output " ++ show i ++ ":"
        print (output :: Vector "Output")
        putStrLn ""

    putStrLn "Note: The neural network weights are randomly initialized, so outputs will vary."

    -- Type safety demonstration
    -- The following lines are intentionally commented out to demonstrate
    -- type safety. Uncommenting them would result in compilation errors.

    -- Attempt invalid matrix multiplication (dimension mismatch)
    -- let invalidMul = matMul (matrix [[1, 2]] :: Matrix "X" "Y") (vector [3, 4] :: Vector "Z")

    -- Attempt to create a neural network with explicit, mismatched types
    -- let invalidNN = initializeNN 2 3 1 :: IO (FeedForwardNN "Input" "Hidden" "Output")

    -- Attempt to use an input vector with incorrect dimensionality
    -- let invalidInput = vector [1.0, 2.0, 3.0] :: Vector "InvalidInput"
    -- let invalidOutput = forwardNN nn invalidInput