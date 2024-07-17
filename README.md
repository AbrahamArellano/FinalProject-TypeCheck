# Typesafe Tensor Operations in Haskell

This project is part of the CS-421 Programming and Compiler course from the University of Illinois Urbana-Champaign.

## Overview

This project implements a subset of typesafe tensor operations as described in the paper "Typesafe Abstractions for Tensor Operations" by Tongfei Chen. It demonstrates the use of Haskell's type system to ensure type safety in tensor operations, particularly focusing on matrix multiplication.

## Project Structure

The project consists of two main Haskell files:

1. `Tensor.hs`: Defines the core tensor types and operations.
2. `Main.hs`: Demonstrates the usage of the tensor operations.

## Features

- Typesafe tensor representations (Scalar, Vector, Matrix)
- Typesafe matrix multiplication
- Basic show instances for demonstration

## Running the Project

To run this project:

1. Ensure you have GHC (Glasgow Haskell Compiler) installed.
2. Clone this repository or copy the `Tensor.hs` and `Main.hs` files.
3. Compile and run the project:

## Extension Ideas

This basic implementation can be extended in several ways:

- Add more tensor operations (addition, subtraction, etc.)
- Implement higher-dimensional tensor operations
- Add support for automatic differentiation
- Implement neural network layers using these tensor operations

## Course Information

- Course: CS-421 Programming and Compiler
- Institution: University of Illinois Urbana-Champaign

## References

Chen, T. (2017). Typesafe Abstractions for Tensor Operations. In Proceedings of the 8th ACM SIGPLAN International Symposium on Scala (SCALA 2017).