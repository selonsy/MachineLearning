#include<stdio.h>
#include<malloc.h>
/*
  美团电话二面
  问题：输入整数N和M，将M表示成N个数相加，找出有多少种表示方法
  思路：
    1.将M表示成X+(M-X)，M-X由（N-1）个数的和累加得来；
    2.递归N次，每一层递归抽离1个数，抽离的数从1-M循环，直到M为0时检测递归深度是否为N，是的话全局计数器加1.
 */
int count = 0;
void addNum(int n, int m, int *arr, int len)
{
  if(n==0&&m==0){
    count++;
    printf("%d", arr[len-1]);
    for(int i=len-2; i>=0; i--)
      printf(" + %d", arr[i]);
    printf("\n");
    return;
  }else if(n==0){
    return;
  }else{
    for(int i=0; m-i>=0; i++){
      arr[n-1] = i;
      addNum(n-1,m-i,arr,len);
    }
  }
}

int main()
{
  int n, m, *arr;
  printf("input n & m: ");
  while(scanf("%d %d", &n, &m)!=EOF){
    if(m<n){
      printf("error: m(%d)<n(%d)\n",m, n);
      continue;
    }
    arr = (int *)malloc(sizeof(int)*n);
    addNum(n, m, arr, n);
    printf("res: %d\n", count);
    count = 0;
    printf("input n & m: ");
    free(arr);
  }
  return 1;
}
