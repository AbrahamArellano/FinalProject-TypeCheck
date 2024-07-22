{-# LANGUAGE DataKinds #-}

module Main where

import Tensor
import Tests
import Control.Monad (forM_)

main :: IO ()
main = do
    putStrLn "Running tests..."
    _ <- runTests
    putStrLn "\nNeural Network Demonstration:"
    nn <- initializeNN 2 3 1
    let inputs = [vector [0.5, 0.8], vector [0.2, 0.9], vector [0.1, 0.3]] :: [Vector "Input"]
    
    putStrLn "Processing inputs through the neural network:"
    forM_ (zip [(1::Int)..] inputs) $ \(i, input) -> do
        let output = forwardNN nn input
        putStrLn $ "Input " ++ show i ++ ":"
        print (input :: Vector "Input")
        putStrLn $ "Output " ++ show i ++ ":"
        print (output :: Vector "Output")
        putStrLn ""

    putStrLn "Note: The neural network weights are randomly initialized, so outputs will vary."

    -- Type safety demonstration (these should not compile if uncommented)
    -- let invalidMul = matMul (matrix [[1, 2]] :: Matrix "X" "Y") (vector [3, 4] :: Vector "Z")
    -- let invalidNN = initializeNN 2 3 1 :: IO (FeedForwardNN "Input" "Hidden" "Output")
    -- let invalidInput = vector [1.0, 2.0, 3.0] :: Vector "InvalidInput"
    -- let invalidOutput = forwardNN nn invalidInput