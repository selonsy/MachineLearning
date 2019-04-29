 #include <stdio.h>
/* 
  题目：求1/1!-1/3!+1/5!-1/7!+.....+(-1)^(n+1)/(2n-1)!
  思路：
    1.累乘，每次结果累加
*/

float add(int n){
    float denominator, sum=0.0;
    for(int i=1; i<=n; i++){
        if(i==1){
            denominator = 1.0;
            printf("1/1");
        }else{
            denominator *= (2*i-1)*(2*i-2);
            printf(" %c 1/%.0f", (i%2 == 0)?'+':'-', denominator);
        }
        sum += ((i%2 == 0)?-1.0:1.0)/denominator;
    }
    return sum;
}
int main(){
    int n;
    printf("Input n(ctrl + c exit): ");
    while(scanf("%d",&n)!=EOF){
        if(n<=0){
            printf("Input Error");
            continue;
        }
        printf(" = %.6f\n",add(n));
        printf("Input n(ctrl + c exit): ");
    }
    return 0;
}
