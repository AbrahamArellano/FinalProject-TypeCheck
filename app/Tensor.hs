{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Tensor where

import GHC.TypeLits
import Data.Kind (Type)
import System.Random (randomRIO)
import Control.Monad (replicateM)

-- Basic Tensor definition
data Tensor (d :: Type) (axes :: [Symbol]) where
  Scalar :: d -> Tensor d '[]
  Vector :: [d] -> Tensor d '[a]
  Matrix :: [[d]] -> Tensor d '[a, b]

-- Type aliases for common tensor types
type Scalar = Tensor Float '[]
type Vector a = Tensor Float '[a]
type Matrix a b = Tensor Float '[a, b]

-- Matrix multiplication
class MatMul a b where
  matMul :: Matrix a b -> Matrix b c -> Matrix a c

instance MatMul a b where
  matMul (Matrix m1) (Matrix m2) = Matrix $ multiply m1 m2
    where
      multiply xs ys = [[sum $ zipWith (*) x y | y <- transpose ys] | x <- xs]
      transpose ([]:_) = []
      transpose x = (map head x) : transpose (map tail x)

-- Helper functions
scalar :: Float -> Scalar
scalar = Scalar

vector :: [Float] -> Vector a
vector = Vector

matrix :: [[Float]] -> Matrix a b
matrix = Matrix

-- Show instances for demonstration
instance Show d => Show (Tensor d '[]) where
  show (Scalar x) = show x

instance Show d => Show (Tensor d '[a]) where
  show (Vector xs) = show xs

instance Show d => Show (Tensor d '[a, b]) where
  show (Matrix xss) = unlines $ map show xss

-- Element-wise operations
class ElementWise (axes :: [Symbol]) where
  elementWise :: (Float -> Float) -> Tensor Float axes -> Tensor Float axes

instance ElementWise '[] where
  elementWise f (Scalar x) = Scalar (f x)

instance ElementWise '[a] where
  elementWise f (Vector xs) = Vector (map f xs)

instance ElementWise '[a, b] where
  elementWise f (Matrix xss) = Matrix (map (map f) xss)

-- Activation functions
relu :: ElementWise axes => Tensor Float axes -> Tensor Float axes
relu = elementWise (\x -> max 0 x)

sigmoid :: ElementWise axes => Tensor Float axes -> Tensor Float axes
sigmoid = elementWise (\x -> 1 / (1 + exp (-x)))

-- Initialize weights and biases
initializeWeights :: Int -> Int -> IO (Matrix a b)
initializeWeights rows cols = Matrix <$> replicateM rows (replicateM cols (randomRIO (-1, 1)))

initializeBiases :: Int -> IO (Vector a)
initializeBiases size = Vector <$> replicateM size (randomRIO (-1, 1))

-- Fully connected layer
data DenseLayer input output = DenseLayer
  { weights :: Matrix input output
  , biases  :: Vector output
  }

forwardDense :: forall input output. (MatMul input output, ElementWise '[output]) => 
                DenseLayer input output -> Vector input -> Vector output
forwardDense (DenseLayer w b) input = 
  let output = matMul w (Matrix [vectorToList input])
  in case output of
       Matrix [result] -> addVectors (Vector result) b

-- Helper function to convert Vector to list
vectorToList :: Vector a -> [Float]
vectorToList (Vector xs) = xs

-- Addition for vectors
addVectors :: Vector a -> Vector a -> Vector a
addVectors (Vector v1) (Vector v2) = Vector $ zipWith (+) v1 v2

-- Simple Feed-forward Neural Network
data FeedForwardNN input hidden output = FeedForwardNN
  { layer1 :: DenseLayer input hidden
  , layer2 :: DenseLayer hidden output
  }

forwardNN :: forall input hidden output. 
             (MatMul input hidden, MatMul hidden output, 
              ElementWise '[hidden], ElementWise '[output]) =>
             FeedForwardNN input hidden output -> Vector input -> Vector output
forwardNN (FeedForwardNN l1 l2) input =
  let hidden = relu $ forwardDense l1 input
      output = sigmoid $ forwardDense l2 hidden
  in output

-- Initialize a Feed-forward Neural Network
initializeNN :: Int -> Int -> Int -> IO (FeedForwardNN a b c)
initializeNN inputSize hiddenSize outputSize = do
  w1 <- initializeWeights hiddenSize inputSize
  b1 <- initializeBiases hiddenSize
  w2 <- initializeWeights outputSize hiddenSize
  b2 <- initializeBiases outputSize
  return $ FeedForwardNN (DenseLayer w1 b1) (DenseLayer w2 b2)