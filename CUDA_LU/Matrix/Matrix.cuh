#pragma once
#include <vector>
#include <utility>

typedef float* raw_matrix;

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
		Alias_Matrix(raw_matrix matrix,size_t full_width,size_t full_height,const std::pair<size_t, size_t>& x_range,const std::pair<size_t, size_t>& y_range);

		__device__ Alias_Matrix operator*(const Alias_Matrix* other_matrix)const;

		__device__ const double& get( size_t x, size_t y) const;
		__device__ double& get_element( size_t x, size_t y);
	};
}
