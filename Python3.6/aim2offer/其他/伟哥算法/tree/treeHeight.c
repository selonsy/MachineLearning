#include<stdio.h>
#include<malloc.h>
/*
    问题：给定节点数N和父亲孩子节点的连接关系，求多叉树树高
    思路：
      1.使用数组存储双亲表示法的二叉树，遍历从数组末尾按照父亲索引网上遍历，找出最大深度。
    用例：
      5
      0,1
      0,2
      1,3
      1,4
      输出结果为3
 */
int main()
{
	int N ;
	int parent, child;
	
	// freopen("in.txt","r",stdin); // test file in.txt

	scanf("%d", &N);
	int *tree = (int *)malloc(sizeof(int *)*N);

	while(scanf("%d %d",&parent, &child)!=EOF){
		printf("input : %d %d\n", parent, child);
		tree[child] = parent;
	}
	tree[0] = -1;
	
	int max_len = 0, len = 0;
	for(int i=N-1; i>=0; i--){
		len = 0;
		for(int j=i; j>=0; j=tree[j]){
			len++;
		}
		if(len>max_len){
			max_len = len;
			for(int j=i; j>=0; j=tree[j]){
				if(j==i)
					printf("%d",j);
				else
					printf("->%d",j);
			}
			printf("\n");
		}
	}
	printf("max_depth: %d\n", max_len);
	return max_len;
}
