#pragma once

#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

namespace CPAR
{
	class Alias_Matrix
	{
	private:
		float* m_matrix;
		unsigned int m_full_width;
		unsigned int m_full_height;
		unsigned int x_phase;
		unsigned int y_phase;

		unsigned int m_width;
		unsigned int m_height;
	public:
		CUDA_CALLABLE_MEMBER Alias_Matrix(float*  matrix,unsigned int full_width,unsigned int full_height,const unsigned int xfirst,const unsigned int x_last,const unsigned int yfirst, const unsigned int ylast)
		{
			m_full_width = full_width;
			m_full_height = full_height;

			x_phase = xfirst;
			y_phase = yfirst;
	
			m_width = m_full_width - x_phase - (m_full_width - x_last - 1);
			m_height = m_full_height - y_phase - (m_full_height - ylast - 1);
		}

		CUDA_CALLABLE_MEMBER void set_data_prt(float* matrix)
		{
			m_matrix = matrix;
		}

		CUDA_CALLABLE_MEMBER const float& get( unsigned int x, unsigned int y) const
		{
			if(y>m_height || x>m_width) return get(0,0);
			return m_matrix[(y + y_phase) * m_full_width + (x + x_phase)];
		}

		CUDA_CALLABLE_MEMBER float& get_element( unsigned int x, unsigned int y)
		{
			if(y>m_height || x>m_width) return get_element(0,0);
			return m_matrix[(y + y_phase) * m_full_width + (x + x_phase)];
		}

		CUDA_CALLABLE_MEMBER void Alias_Matrix::print() const
		{
			for (size_t i = 0; i < m_height; ++i)
			{
				for (size_t j = 0; j < m_width; ++j)
				{
					printf(" %f ", get(j, i));
				}
				printf("\n");
			}
		}
	};
}
