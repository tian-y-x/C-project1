#include <iostream>
#include <ctime>
#include <cmath>
#include <chrono>
#include <string>
#include <omp.h>
#include <immintrin.h>
#pragma optimize (3)
#define Timecost(x,y) cout<<chrono::duration_cast<chrono::milliseconds>(x - y).count() << "毫秒" << endl;

using namespace std;
//using namespace literals;



#define Big
//#define input

#ifdef Big
const int maxn = 10000;
const int Arow = maxn;
const int Acolumn = maxn;
const int Bcolumn = maxn;
struct BigData
{
	int row,column,count;
	float* M;
	BigData(int row, int column)
	{
		this->row = row, this->column = column;
		count = row * column;
		M = new float[row * column];
		if (!M)
			exit(-1);
	}
	~BigData()
	{
		delete[] this->M;
	}
	void Init()
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < column; j++)
			{
				M[i * this->column + j] = float(1);
			}
		}
	}
};
#endif 
#ifdef input
struct Martrix
{
	size_t column, row, count;
	float* M;
	string str;
	Martrix()
	{
		column = row = count = 0;
		M = (float*)malloc(10000 * sizeof(float));
	}
	void equal(const Martrix& b)
	{
		if (this->column != b.row)
			cout << "\n两矩阵不可相乘\na.row=" << this->row<<" a.column=" << this->column 
			<< "\nb.row=" << b.row << " b.column=" << b.column << endl, exit(-1);
		else
		{
			cout << "可以合法进行矩阵乘法" << endl;
			cout << "a.row a.column分别为" << this->row <<'\t'<< this->column << endl;
			cout << "b.row b.column分别为" << b.row <<'\t'<< b.column << endl;
		}
		return;
	}
	void in()
	{
		cout << "读入新矩阵,单独一行的end作为结束符\n";
		float temp;
		while (getline(cin, str)&&str!="end")//一行行读入,分别对每一行进行读入、检查
		{
			
			size_t pos = 0, tpos = 0;
			long long curColumn = 0;//记录当前行的个数，便于检查读入是否合法
			char* p = &(str[0]);
			while (pos != str.length())
			{
				while (str[pos] == ' ')
					pos++;

				if (str[pos] != '-' && str[pos] != '+' && (str[pos] < '0' || str[pos]>'9') && (str[pos] != ' '))
					std::cout << "输入不合法" << endl, exit(-1);
				if (this->count % 10000 == 0)
					realloc(M, 10000 + this->count);
				temp = stof(p + pos, &tpos);
				this->M[this->count++] = temp;
				curColumn++;
				pos += tpos;
				if (str[pos] == '-' || str[pos] == '+')
					std::cout << "输入不合法" << endl, exit(-1);  //避免出现形如-123-2被判断为合法
				
			}
			this->row++;
			if(abs(this->count*1.0/curColumn-row)>0.000001)
				std::cout << "输入不合法" << endl, exit(-1);  //避免出现形如-123-2被判断为合法
		}
		this->column = this->count / row;
		cout << "该矩阵读入完毕，输入合法\n\n";
	}
	
};
#endif

float* p, * q, * m, * n;

/**************************SIMD指令集***************************/



/**************************SIMD指令集***************************/






int i, j, k;
int main()
{
	

	auto startio = chrono::steady_clock::now();

	/*********初始化*********/
#ifdef Big
	/*BigData M1(Arow,Acolumn), M2(Acolumn,Bcolumn), M3(Arow,Bcolumn);
	M1.Init();	
	M2.Init();	
	M3.Init();
	m = M1.M;
	n = M2.M;	*/
	//memset(M3.M, 0, M1.row * M2.column * sizeof(float));

	
	float(*vec1)[maxn] = new float[maxn][maxn];
	float(*vec2)[maxn] = new float[maxn][maxn];
	float(*vec)[maxn] = new float[maxn][maxn];
	for (i = 0; i < maxn; i++)
	{
		for (j = 0; j < maxn; j++)
		{
			vec1[i][j] = 1;
			vec2[i][j] = 1;
			vec[i][j] = 0;
		}
	}
	__m256 a, b;
	__m256 c = _mm256_setzero_ps();
	float sum[8];
	/*for (i = 0; i < maxn; i++)
	{
		for (j = 0; j < maxn; j++)
		{
			vec1[i][j] = M1.M[i * M2.column + j];
			vec2[i][j] = M2.M[i * M2.column + j];
		}
	}*/
	auto start = chrono::steady_clock::now();
	cout << "The IO cost:";  Timecost(start, startio);

	///************计算***************/


	/*for (i = 0; i < M1.row; i++)
	{
		for (j = 0; j < M2.column; j++)
		{

			for (k = 1; k <= M1.column; k++)
			{
				M3.M[i * M1.column + j] += M1.M[i * M1.column + k - 1] * M2.M[(k - 1) * M2.column + j];
			}
		}
	}*/
#pragma omp parallel for collapse(2) schedule(dynamic) private(i, j, k) shared(vec1, vec2, vec)
	for (i = 0; i < maxn; i++)
	{
		for (j = 0; j < maxn; j++)
		{
			for (k = 0; k < maxn; k++)
			{
				vec[i][j] += vec1[i][k] * vec2[k][j];
				a = _mm256_loadu_ps(vec1[i] + k);
				b = _mm256_loadu_ps(vec2[i] + k);
				c = _mm256_mul_ps(a, b);

				_mm256_store_ps(&vec[i][j], c);
			}
		}
	}
	//vec[i][j] += vec1[i][k] * vec2[k][j];

	//for (int i = 0; i < M1.row; i++)
	//{
	//	for (int j = 0; j < M2.column; j++)
	//	{
	//			cout << vec[i][j];
	//	}
	//}

	//for (int i = 0; i < M1.row; i++)
	//{
	//	for (int j = 0; j < M2.column; j++)
	//	{
	//		if (j != M2.column - 1)
	//			cout << M3.M[i * M2.column + j] << " ";
	//		else
	//			cout << M3.M[i * M2.column + j] << endl;
	//	}
	//}


	/************计算***************/
	auto end = chrono::steady_clock::now();
	cout << "The calculate cost:";  Timecost(end, start);
	//delete[] vec1; delete[]vec2; delete[]vec;


	
//良哥代码
//	for (i = 0; i < maxn; ++i)
//	{
//		for (j = 0; j < maxn; ++j)
//		{
//			vec1[i][j] = 1;
//			vec2[i][j] = 1;
//		}
//	}
//	clock_t s1, t1, s2, t2;
//	//printf("--------------before parallel compute---------------\n");
//	
//	//s1 = clock();
//	//for (i = 0; i < maxn; ++i)
//	//{
//	//	for (j = 0; j < maxn; ++j)
//	//	{
//	//		for (k = 0; k < maxn; ++k)
//	//		{
//	//			vec[i][j] += (vec1[i][k] * vec2[k][j]);
//	//		}
//	//	}
//	//}
//	//
//	//t1 = clock();
//	//printf("----------------used time = %d ms-----------------\n", t1 - s1);
//
//
//	printf("--------------enter parallel compute---------------\n");
//	s2 = clock();
//
//#pragma omp parallel for collapse(2) schedule(dynamic) private(i, j, k) shared(vec1, vec2, vec)
//	for (i = 0; i < maxn; ++i)
//	{
//		for (j = 0; j < maxn; ++j)
//		{
//			for (k = 0; k < maxn; ++k)
//			{
//				vec[i][j] += (vec1[i][k] * vec2[k][j]);
//			}
//		}
//	}
//	t2 = clock();
//	printf("----------------used time = %d ms-----------------\n", t2 - s2);
//
//	//printf("\n----------------the speedup ratio = %lf---------------\n", 1.0 * (t1 - s1) / (t2 - s2));
//




#endif


#ifdef input
	Martrix a, b;
	a.in();
	b.in();
	a.equal(b);

	float *c=new float [a.row*b.column];
	memset(c, 0, a.row*b.column*sizeof(float));

	//auto start = chrono::steady_clock::now();

	for (int i = 0; i < a.row; i++) 
		for (int j = 0; j < b.column; j++) 
			for (int k = 1; k <= a.column; k++) 
				c[i*b.column+j] += a.M[i * a.column + k - 1] * b.M[(k - 1) * b.column + j];

	for (int i = 0; i < a.row; i++) 
	{
		for (int j = 0; j < b.column; j++) 
		{
			if (j != b.column - 1) 
				cout << c[i * b.column + j] << " ";
			else 
				cout << c[i * b.column + j] << endl;
		}
	}
	


	//auto end = chrono::steady_clock::now();
	//cout << "The calculate cost:";  Timecost(end, start);


#endif
	return 0;
}
