#include<iostream>
#include<limits.h>
#include<malloc.h>
#include<vector>
#include<map>
/*
  大疆2017秋招研发岗
  问题：m个物品散落在n个地点，这n个地点间有L个路径相连，小明需要从起始位置x尽快去到丢失最重要物品的位置y,路上尽量把能捡起的东西都捡起来，但是不能捡相同编号的物品，求小明从x到y的最短路径长度和最短路径上能捡到的物品个数。
  思路：
    1.dijkstra算法求得单源最短路径，使用数组表示的孩子父亲表示法多叉树保存路径，最后从目标节点回溯到起始节点，并用哈希表保存存放的物品编号。

 */
using namespace std;

struct Node{
	vector<int> stuff;
};

int ** initNeighbourMatrix(int n){
	/* 初始化邻接矩阵 */
	int **matrix;
	matrix = (int **)malloc(sizeof(int *)*n);
	for(int i=0; i<n; i++){
		matrix[i] = (int *)malloc(sizeof(int)*n); // 动态申请一维数组
	}

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			if(i==j)
				matrix[i][j] = 0;
			else
				matrix[i][j] = INT_MAX;
		}
	}

	return matrix;
}

int main(){
	int n, L, m;

	cin >> n; // 节点个数
	cin >> L; // 路径个数
	cin >> m; // 物品个数
	
	int **matrix; // 邻接矩阵
	matrix = initNeighbourMatrix(n); // 初始化邻接矩阵

	int s, e, w; // 路径起点、路径终点、路径权重
	for(int i=0; i<L; i++){
		cin >> s;
		cin >> e;
		cin >> w;
		matrix[s-1][e-1] = w;
		matrix[e-1][s-1] = w;		
	}

	int index, node;
	Node *nodeStuff; // 节点中掉落需要捡起的物品
	nodeStuff = (Node *)malloc(sizeof(Node)*m);

	for(int i=0; i<m; i++){
		cin >> index; 
		cin >> node;
		nodeStuff[node-1].stuff.push_back(index);
	}

	int start, target;
	cin >> target; // 终点
	cin >> start;  // 起始点
	start -= 1;	// 使节点编号直接使用
	target -= 1;	
	
	int *U, *S;
	U = (int *)malloc(sizeof(int)*n); // 起始点start到其他节点的当前距离，未加入最短路径的节点集
	S = (int *)malloc(sizeof(int)*n); // 起始点start到其他节点的最短路径，加入最短路径的节点集
	for(int j=0; j<n; j++){
		U[j] = matrix[start][j]; // 把start的邻接点加入到U集合中；
		S[j] = INT_MAX;			
	}
	U[start] = -1;
	S[start] = 0;

	// 数组表示n叉树孩子父亲表示法，以出发节点为父亲，终点为其中的叶子节点，保存终点到起始点的路径
	int *parent; 
	parent = (int *)malloc(sizeof(int)*n);
	parent[start] = -1;
	for(int i=0; i<n; i++){
		if(matrix[start][i] < INT_MAX && matrix[start][i] > 0)
			parent[i] = start; // 初始化start的直连节点的父亲表示
	}

	int minNode, minValue; // 记录U集合中最短路径节点和路径大小
	//int lastNode = start; // 保存当前节点的父亲节点，刚加入S的节点作为父亲
	int count = 0; // 记录添加最短路径的次数
	while(count < n-1){
		// 运行n-1次，总共有n-1个节点从集合U加入到集合S
		for(int i=0; i<n; i++){
			if(minValue > U[i] && U[i] > 0 ){
				// 找到U中start能到达的最小距离节点i设置为minNode
				minValue = U[i];
				minNode = i;
			}
		}
		U[minNode] = -1;
		S[minNode] = minValue;
		minValue = INT_MAX;
		for(int i=0; i<n; i++){
			if(matrix[minNode][i] < INT_MAX && S[minNode] + matrix[minNode][i] < U[i]){
				// 刚加入S集合的节点minNode到节点i的距离不为无穷大 
				// 且 start节点到minNode的距离加上minNode到i的距离比当前start到i的距离小
				U[i] = S[minNode] + matrix[minNode][i];
				parent[i] = minNode; // 若存在minNode使start->minNode->i路径最短，则更新minNode为i的父亲节点
			}
		}
		count++;
	}

	int stuff; // 加入最短路径过程中
	map<int, int> package; // 小明的背包，用来捡起散落的物品
	map<int, int>::iterator iter;
	for(int i=target; i>=0; i = parent[i]){
		for(int s=0; s<nodeStuff[i].stuff.size(); s++){
			stuff = nodeStuff[i].stuff[s];
			iter = package.find(stuff);
			if(iter == package.end()){
				package[stuff] = 1;
			}
		}
	}
	cout << S[target] << " " << package.size() << endl;

	return 0;
}
