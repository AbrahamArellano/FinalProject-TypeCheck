{-# LANGUAGE DataKinds #-}

module Tests where

import Test.HUnit
import Tensor

-- Basic Tensor Creation Tests
-- These tests verify the correct creation of scalar, vector, and matrix tensors
testScalarCreation :: Test
testScalarCreation = TestCase (assertEqual "Scalar creation" 5 (getScalarValue (scalar 5)))

testVectorCreation :: Test
testVectorCreation = TestCase (assertEqual "Vector creation" [1,2,3] (getVectorValue (vector [1,2,3])))

testMatrixCreation :: Test
testMatrixCreation = TestCase (assertEqual "Matrix creation" [[1,2],[3,4]] (getMatrixValue (matrix [[1,2],[3,4]])))

-- Matrix Multiplication Test
-- This test checks the correctness of matrix-vector multiplication
testMatrixMultiplication :: Test
testMatrixMultiplication = TestCase (assertEqual "Matrix-vector multiplication" 
    [19.0, 43.0] 
    (getVectorValue (matMul (matrix [[1.0,2.0],[3.0,4.0]]) (vector [5.0,6.0]))))

-- Neural Network Forward Pass Test
-- This test ensures that a neural network can perform a forward pass
-- and produce output of the expected dimension
testNeuralNetworkForwardPass :: Test
testNeuralNetworkForwardPass = TestCase $ do
    nn <- initializeNN 2 3 1
    let input = vector [0.5, 0.8] :: Vector "Input"
    let output = forwardNN nn input
    assertEqual "NN output dimension" 1 (length $ getVectorValue output)

-- Large-scale Matrix Multiplication Test
-- This test verifies that the implementation can handle larger matrices
-- and checks the dimension of the output
testLargeMatrixMultiplication :: Test
testLargeMatrixMultiplication = TestCase $ do
    let m = matrix [[fromIntegral i :: Float | i <- [1..100]] | _ <- [1..100]] :: Matrix "A" "B"
    let v = vector [fromIntegral i :: Float | i <- [1..100]] :: Vector "A"
    let result = matMul m v
    assertEqual "Large matrix-vector multiplication" 100 (length $ getVectorValue result)

-- Helper functions to extract values from tensors
getScalarValue :: Scalar -> Float
getScalarValue (Scalar x) = x

getVectorValue :: Vector a -> [Float]
getVectorValue (Vector xs) = xs

getMatrixValue :: Matrix a b -> [[Float]]
getMatrixValue (Matrix xss) = xss

-- Aggregation of all tests
tests :: Test
tests = TestList [
    TestLabel "Scalar Creation" testScalarCreation,
    TestLabel "Vector Creation" testVectorCreation,
    TestLabel "Matrix Creation" testMatrixCreation,
    TestLabel "Matrix Multiplication" testMatrixMultiplication,
    TestLabel "Neural Network Forward Pass" testNeuralNetworkForwardPass,
    TestLabel "Large Matrix Multiplication" testLargeMatrixMultiplication
    ]

-- Function to run all tests
runTests :: IO Counts
runTests = runTestTT tests