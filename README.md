# High Power Computing: Sparse Matrix-Vector Multiplication with BCSR Format

This repository focuses on the Sparse Matrix-Vector Multiplication (SpMV) problem, specifically exploring the use of the Block Compressed Sparse Row (BCSR) format with MPI (Message Passing Interface). SpMV is a critical task in the realm of High-Performance Computing (HPC), and this project aims to provide efficient parallel algorithms and implementations for SpMV using the BCSR format with MPI.

## **Overview**

The SpMV problem is practiced in this project under the assumptions that the Sparse Matrix is a Symmetric Nine-Banded matrix and that it is stored in the Block Compressed Sparse Row (BCSR) format. The BCSR format enables efficient storage and processing of sparse matrices, while MPI allows for distributed computing across multiple nodes or processors.

The project includes the following components:

1. Code Examples: The repository provides code examples and implementations showcasing the SpMV algorithm using the BCSR format with MPI. Each step of the algorithm is explained within the code, making it easier to understand the parallelization techniques used.
2. Parallelization Strategy: The project employs an MPI-based parallelization strategy to distribute the computation of the SpMV algorithm across multiple processes. This allows for efficient utilization of available computing resources and improved performance.
3. Performance Optimization: Various techniques and optimizations are explored to enhance the performance of the SpMV algorithm with the BCSR format and MPI. These optimizations aim to reduce communication overhead, load balance the computation, and improve scalability.
4. Benchmarks: The project includes benchmarks and performance evaluations of the parallelized SpMV algorithm using the BCSR format with MPI. These benchmarks provide insights into the algorithm's efficiency, scalability, and potential bottlenecks when running on distributed computing systems.

## **ITU UHEM Supercomputer**

To facilitate the development and testing of the parallelized SpMV algorithm, the project utilizes the ITU UHEM supercomputer. This powerful computing resource offers substantial computational power and resources, enabling researchers to tackle complex HPC problems effectively using MPI-based parallelization.

## **Blog**

For more detailed explanations, discussions, and insights into the algorithm's implementation and performance, please refer to the accompanying blog. The blog serves as a comprehensive guide to understanding and applying SpMV techniques with the BCSR format and MPI. It delves into theoretical concepts, algorithmic optimizations, experimental results, and practical considerations in the realm of Sparse Matrix-Vector Multiplication in HPC.

Visit the blog at **[link-to-blog](https://cagnurt.github.io/projects/proj-2)** to explore the fascinating world of Sparse Matrix-Vector Multiplication in HPC, and parallelization using MPI, and to enhance your understanding and implementations of the SpMV algorithm with the BCSR format.

## **Contributions**

Contributions to this repository are welcome. If you have any suggestions, improvements, or bug fixes, please feel free to submit a pull request. Together, we can advance the field of Sparse Matrix-Vector Multiplication in HPC, parallel computing with MPI, and improve the performance of the SpMV algorithm using the BCSR format.

## **License**

This project is licensed under the **[MIT License](https://chat.openai.com/LICENSE)**. Feel free to use the code, algorithms, and resources provided in this repository in accordance with the license terms.