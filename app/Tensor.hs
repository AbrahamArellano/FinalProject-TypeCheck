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

-- Tensor data type with type-level axes
data Tensor (d :: Type) (axes :: [Symbol]) where
  Scalar :: d -> Tensor d '[]
  Vector :: [d] -> Tensor d '[a]
  Matrix :: [[d]] -> Tensor d '[a, b]

-- Show instances for better readability
instance Show d => Show (Tensor d '[]) where
  show (Scalar x) = show x

instance Show d => Show (Tensor d '[a]) where
  show (Vector xs) = show xs

instance Show d => Show (Tensor d '[a, b]) where
  show (Matrix xss) = unlines $ map show xss

-- Type aliases for common tensor types
type Scalar = Tensor Float '[]
type Vector a = Tensor Float '[a]
type Matrix a b = Tensor Float '[a, b]

-- Matrix multiplication typeclass
class MatMul a b where
  matMul :: Matrix a b -> Vector a -> Vector b

instance MatMul a b where
  matMul (Matrix m) (Vector v) = Vector $ map (sum . zipWith (*) v) m

-- Helper functions for tensor creation
scalar :: Float -> Scalar
scalar = Scalar

vector :: [Float] -> Vector a
vector = Vector

matrix :: [[Float]] -> Matrix a b
matrix = Matrix

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

-- Neural network components
data DenseLayer input output = DenseLayer
  { weights :: Matrix input output
  , biases  :: Vector output
  }

forwardDense :: (MatMul input output) => 
                DenseLayer input output -> Vector input -> Vector output
forwardDense (DenseLayer w b) input = addVectors (matMul w input) b

addVectors :: Vector a -> Vector a -> Vector a
addVectors (Vector v1) (Vector v2) = Vector $ zipWith (+) v1 v2

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

-- Initialization functions for neural network components
initializeWeights :: Int -> Int -> IO (Matrix a b)
initializeWeights rows cols = Matrix <$> replicateM rows (replicateM cols (randomRIO (-1, 1)))

initializeBiases :: Int -> IO (Vector a)
initializeBiases size = Vector <$> replicateM size (randomRIO (-1, 1))

initializeNN :: Int -> Int -> Int -> IO (FeedForwardNN a b c)
initializeNN inputSize hiddenSize outputSize = do
  w1 <- initializeWeights hiddenSize inputSize
  b1 <- initializeBiases hiddenSize
  w2 <- initializeWeights outputSize hiddenSize
  b2 <- initializeBiases outputSize
  return $ FeedForwardNN (DenseLayer w1 b1) (DenseLayer w2 b2)