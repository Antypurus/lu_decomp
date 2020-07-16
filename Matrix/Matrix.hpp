#pragma once
#include <vector>
#include <utility>

typedef double* raw_matrix;
typedef double** raw_2d_matrix;
typedef std::vector<std::vector<double>> vector_2d_matrix;

namespace CPAR
{

	class Alias_Matrix
	{
	private:
		raw_matrix m_matrix;
		size_t full_width;
		size_t full_height;
		size_t x_phase;
		size_t y_phase;

		size_t m_width;
		size_t m_height;
	public:
		Alias_Matrix(raw_matrix matrix, size_t full_width, size_t full_height, const std::pair<size_t, size_t>& x_range, const std::pair<size_t, size_t>& y_range);

		Alias_Matrix operator*(const Alias_Matrix& other_matrix)const;

		const double& get(size_t x, size_t y) const;
		double& get_element(size_t x, size_t y);

		void print() const;
	};

	class Matrix
	{
	private:
		raw_matrix m_matrix = NULL;
		size_t m_width;
		size_t m_height;
	public:
		Matrix(const raw_matrix& matrix, const size_t width, const size_t height);
		Matrix(const raw_2d_matrix& matrix, const size_t width, const size_t height);
		Matrix(const vector_2d_matrix& matrix);
		Matrix(const Matrix& matrix);
		~Matrix();

		Alias_Matrix generate_aliasing_matrix(const std::pair<size_t, size_t>& x_range, const std::pair<size_t, size_t>& y_range) const;

		double& getElement(const size_t x, const size_t y);
		const double& get(const size_t x, const size_t y)const;
		void print()const;

		Matrix operator=(const Matrix& matrix);

		static Matrix sequential_LU_decomposition(const Matrix& matrix);
		static Matrix parallel_LU_decomposition(const Matrix& matrix);
		static Matrix block_reduced_base_sequential_LU_decomposition(const Matrix& matrix, const size_t block_size);
		static Matrix block_LU_decomposition(const Matrix& matrix, size_t block_size);
	};
}
