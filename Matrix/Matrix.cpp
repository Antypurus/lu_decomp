#include "Matrix.hpp"
#include <cstring>
#include <exception>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <new>
#include <ctime>

namespace CPAR
{
	// Alias Matrix Code Start //
	Alias_Matrix::Alias_Matrix(raw_matrix matrix, size_t full_width, size_t full_height,
		const std::pair<size_t, size_t>& x_range, const std::pair<size_t, size_t>& y_range)
		:m_matrix(matrix), full_width(full_width), full_height(full_height)
	{
		this->x_phase = x_range.first;
		this->y_phase = y_range.first;

		this->m_width = full_width - x_phase - (full_width - x_range.second - 1);
		this->m_height = full_height - y_phase - (full_height - y_range.second - 1);
	}

	Alias_Matrix Alias_Matrix::operator*(const Alias_Matrix& other_matrix)const
	{
		using namespace  std;
		size_t block_size = this->m_width;
		Alias_Matrix res(this->m_matrix, this->full_width, this->full_height,
			std::make_pair(block_size, this->full_width - 1),
			std::make_pair(block_size, this->full_height - 1));
		for (size_t y = 0; y < this->m_height; y++)
		{
			for (size_t x = 0; x < other_matrix.m_width; x++)
			{
				double temp = 0;
				for (size_t k = 0; k < this->m_width; k++)
				{
					temp += this->get(k, y) * other_matrix.get(x, k);
				}
				res.get_element(x, y) -= temp;
			}
		}
		return res;
	}

	const double& Alias_Matrix::get(size_t x, size_t y) const
	{
#ifndef NDEBUG
		if (x >= this->m_width || x < 0)
		{
			throw std::runtime_error("X coordinate out of bounds");
		}
		if (y >= this->m_height || y < 0)
		{
			throw std::runtime_error("Y coordinate our of bounds");
		}
#endif
		return this->m_matrix[(y + y_phase) * this->full_width + (x + x_phase)];
	}

	double& Alias_Matrix::get_element(size_t x, size_t y)
	{
#ifndef NDEBUG
		if (x >= this->m_width || x < 0)
		{
			throw std::runtime_error("X coordinate out of bounds");
		}
		if (y >= this->m_height || y < 0)
		{
			throw std::runtime_error("Y coordinate our of bounds");
		}
#endif
		return this->m_matrix[(y + y_phase) * this->full_width + (x + x_phase)];
	}

	void Alias_Matrix::print() const
	{
		for (size_t i = 0; i < this->m_height; ++i)
		{
			for (size_t j = 0; j < this->m_width; ++j)
			{
				printf(" %f ", this->get(j, i));
			}
			printf("\n");
		}
	}
	// Alias Matrix Code End//

	// Matrix Code Start//
	Matrix::Matrix(const raw_matrix& matrix, const size_t width, const size_t height)
		:m_width(width), m_height(height)
	{
		this->m_matrix = new double[width * height];
		std::memcpy(&this->m_matrix[0], &matrix[0], width * height * sizeof(raw_matrix));
	}

	Matrix::Matrix(const raw_2d_matrix& matrix, const size_t width, const size_t height)
		:m_width(width), m_height(height)
	{
		this->m_matrix = new double[width * height];
		for (size_t i = 0; i < height; ++i)
		{
			memmove(&this->m_matrix[i * width], matrix[i], width * sizeof(raw_matrix));
		}
	}

	Matrix::Matrix(const vector_2d_matrix& matrix)
		:m_width(matrix.size()), m_height(matrix[0].size())
	{
		this->m_matrix = new double[this->m_width * this->m_height];
		for (size_t i = 0; i < this->m_height; ++i)
		{
			for (size_t j = 0; j < this->m_width; ++j)
			{
				this->getElement(j, i) = matrix[j][i];
			}
		}
	}

	Matrix::Matrix(const Matrix& matrix)
		:m_width(matrix.m_width), m_height(matrix.m_height)
	{
		const size_t matrixSize = matrix.m_width * matrix.m_height;
		m_matrix = new double[matrixSize];
		std::memcpy(this->m_matrix, matrix.m_matrix, this->m_width * this->m_height * sizeof(raw_matrix));
	}

	Matrix::~Matrix()
	{
		delete this->m_matrix;
	}

	Alias_Matrix Matrix::generate_aliasing_matrix(const std::pair<size_t, size_t>& x_range,
		const std::pair<size_t, size_t>& y_range) const
	{
		return Alias_Matrix{ this->m_matrix, this->m_width, this->m_height, x_range, y_range };
	}

	double& Matrix::getElement(const size_t x, const size_t y)
	{
#ifndef NDEBUG
		if (x >= this->m_width || x < 0)
		{
			throw std::runtime_error("X coordinate out of bounds");
		}
		if (y >= this->m_height || y < 0)
		{
			throw std::runtime_error("Y coordinate our of bounds");
		}
#endif
		return this->m_matrix[y * this->m_width + x];
	}

	const double& Matrix::get(const size_t x, const size_t y) const
	{
#ifndef NDEBUG
		if (x >= this->m_width || x < 0)
		{
			throw std::runtime_error("X coordinate out of bounds");
		}
		if (y >= this->m_height || y < 0)
		{
			throw std::runtime_error("Y coordinate out of bounds");
		}
#endif
		return this->m_matrix[y * this->m_width + x];
	}

	void Matrix::print() const
	{
		for (size_t i = 0; i < this->m_height; ++i)
		{
			for (size_t j = 0; j < this->m_width; ++j)
			{
				printf(" %f ", this->get(j, i));
			}
			printf("\n");
		}
	}

	Matrix Matrix::sequential_LU_decomposition(const Matrix& matrix)
	{
		Matrix result(matrix);
		const size_t matrix_size = result.m_width;

		for (size_t k = 0; k < matrix_size; ++k)
		{
			for (size_t j = k; j < matrix_size; ++j)
			{
				double sum = 0.;
				for (size_t p = 0; p < k; ++p)
				{
					sum += result.get(p, k) * result.get(j, p);
				}
				result.getElement(j, k) = result.get(j, k) - sum;
			}

			for (size_t i = k + 1; i < matrix_size; ++i)
			{
				double sum = 0.;
				for (size_t p = 0; p < k; ++p)
				{
					sum += result.get(p, i) * result.get(k, p);
				}
				result.getElement(k, i) = (result.get(k, i) - sum) / result.get(k, k);
			}
		}

		return result;
	}

	Matrix Matrix::parallel_LU_decomposition(const Matrix& matrix)
	{
		Matrix result(matrix);
		const size_t matrix_size = result.m_width;
		long long int k;

		#pragma omp parallel private(k) num_threads(2)
		for (k = 0; k < matrix_size; ++k)
		{
			#pragma omp for
			for (long long int j = k; j < matrix_size; ++j)
			{
				double sum = 0.;
				for (long long int p = 0; p < k; ++p)
				{
					sum += result.get(p, k) * result.get(j, p);
				}
				result.getElement(j, k) = result.get(j, k) - sum;
			}

			//All threads are synced here, because the value of the pivot, that will be used in the next for loop, is changed by the previous for loop

			#pragma omp for
			for (long long int i = k + 1; i < matrix_size; ++i)
			{
				double sum = 0.;
				for (long long int p = 0; p < k; ++p)
				{
					sum += result.get(p, i) * result.get(k, p);
				}
				result.getElement(k, i) = (result.get(k, i) - sum) / result.get(k, k);
			}
		}

		return result;
	}

	Matrix Matrix::block_reduced_base_sequential_LU_decomposition(const Matrix& matrix, const size_t block_size)
	{
		Matrix result(matrix);
		const size_t matrix_size = result.m_width;

		for (size_t k = 0; k < block_size; ++k)
		{
			for (size_t j = k; j < matrix_size; ++j)
			{
				double sum = 0.;
				for (size_t p = 0; p < k; ++p)
				{
					sum += result.get(p, k) * result.get(j, p);
				}
				result.getElement(j, k) = result.get(j, k) - sum;
			}
			for (size_t i = k + 1; i < matrix_size; ++i)
			{
				double sum = 0.;
				for (size_t p = 0; p < k; ++p)
				{
					sum += result.get(p, i) * result.get(k, p);
				}
				result.getElement(k, i) = (result.get(k, i) - sum) / result.get(k, k);
			}
		}

		return result;
	}

	Matrix Matrix::block_LU_decomposition(const Matrix& matrix, size_t block_size)
	{
		auto result = Matrix::block_reduced_base_sequential_LU_decomposition(matrix, block_size);

		//apply multiplication of lower and upper matrixes to get final result
		auto upper_prime_matrix = result.generate_aliasing_matrix(std::make_pair(block_size, result.m_width - 1), std::make_pair(0, block_size - 1));
		auto lower_prime_matrix = result.generate_aliasing_matrix(std::make_pair(0, block_size - 1), std::make_pair(block_size, result.m_height - 1));
		lower_prime_matrix * upper_prime_matrix;

		return result;
	}
}
