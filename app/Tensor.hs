{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module Tensor where

import GHC.TypeLits
import Data.Kind (Type)

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