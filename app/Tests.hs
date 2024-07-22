{-# LANGUAGE DataKinds #-}

module Tests where

import Test.HUnit
import Tensor

testScalarCreation :: Test
testScalarCreation = TestCase (assertEqual "Scalar creation" 5 (getScalarValue (scalar 5)))

testVectorCreation :: Test
testVectorCreation = TestCase (assertEqual "Vector creation" [1,2,3] (getVectorValue (vector [1,2,3])))

testMatrixCreation :: Test
testMatrixCreation = TestCase (assertEqual "Matrix creation" [[1,2],[3,4]] (getMatrixValue (matrix [[1,2],[3,4]])))

testMatrixMultiplication :: Test
testMatrixMultiplication = TestCase (assertEqual "Matrix-vector multiplication" 
    [19.0, 43.0] 
    (getVectorValue (matMul (matrix [[1.0,2.0],[3.0,4.0]]) (vector [5.0,6.0]))))

testNeuralNetworkForwardPass :: Test
testNeuralNetworkForwardPass = TestCase $ do
    nn <- initializeNN 2 3 1
    let input = vector [0.5, 0.8] :: Vector "Input"
    let output = forwardNN nn input
    assertEqual "NN output dimension" 1 (length $ getVectorValue output)

testLargeMatrixMultiplication :: Test
testLargeMatrixMultiplication = TestCase $ do
    let m = matrix [[fromIntegral i :: Float | i <- [1..100]] | _ <- [1..100]] :: Matrix "A" "B"
    let v = vector [fromIntegral i :: Float | i <- [1..100]] :: Vector "A"
    let result = matMul m v
    assertEqual "Large matrix-vector multiplication" 100 (length $ getVectorValue result)

getScalarValue :: Scalar -> Float
getScalarValue (Scalar x) = x

getVectorValue :: Vector a -> [Float]
getVectorValue (Vector xs) = xs

getMatrixValue :: Matrix a b -> [[Float]]
getMatrixValue (Matrix xss) = xss

tests :: Test
tests = TestList [
    TestLabel "Scalar Creation" testScalarCreation,
    TestLabel "Vector Creation" testVectorCreation,
    TestLabel "Matrix Creation" testMatrixCreation,
    TestLabel "Matrix Multiplication" testMatrixMultiplication,
    TestLabel "Neural Network Forward Pass" testNeuralNetworkForwardPass,
    TestLabel "Large Matrix Multiplication" testLargeMatrixMultiplication
    ]

runTests :: IO Counts
runTests = runTestTT tests