#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<string.h>
#include<fstream>
#define MAX 200000
int assembleLineSequence(int *e, int t[][MAX], int a[][MAX],int *x, int n)
{
    int *total[2];    //记录到达每个station最小的时间花费
    total[0] = (int *)malloc(sizeof(int)*n);
    total[1] = (int *)malloc(sizeof(int)*n);
    total[0][0] = e[0] + a[0][0];
    total[1][0] = e[1] + a[1][0];
    for(int i=1; i<n; i++){
        if(total[0][i-1] + a[0][i] < total[1][i-1] + t[1][i-1] + a[0][i])
            total[0][i] = total[0][i-1] + a[0][i];
        else
            total[0][i] = total[1][i-1] + t[1][i-1] + a[0][i];
        if(total[1][i-1] + a[1][i] < total[0][i-1] + t[0][i-1] + a[1][i])
            total[1][i] = total[1][i-1] + a[1][i];
        else
            total[1][i] = total[0][i-1] + t[0][i-1] + a[1][i];
    }
    if(total[0][n-1] + x[0] < total[1][n-1] + x[1])
        return total[0][n-1] + x[0];
    return total[1][n-1] + x[1];
}

int main()
{
    FILE *fp;
    int result;
    clock_t start, end;    
    for(int i=1; i<=10; i++){
        start = clock();
        char data[20] = "第";　　//给足够大的长度才不会出现段错误
        char s[2];
        sprintf(s,"%d",i);
        strcat(data,s);
        strcat(data, "组.txt");
        if((fp=fopen(data,"r"))==NULL){
            printf("Cannot open file strike any key exit!");
            return -1;
        }
        bool flag = false;    //记录当前字符是否与上一字符拼接成新数字
        int num,j=0,n=0,line=1;    //n记录数据个数
        int t[2][MAX],a[2][MAX];
        int e[2] = {0,0},x[2] = {0,0};
        char nc;
        while(!feof(fp)){
            nc = fgetc(fp);
            if(isdigit(nc)){
                if(flag == true){
                    num = num * 10 + (nc - '0');
                }else{
                    num = nc - '0';
                }
                flag = true;
            }else {                
                if(nc=='\n'){
                    n = j;
                    j = 0;                    
                    line++;
                }
                if(nc == ' '){
                    switch(line){
                        case 1:
                            t[0][j++] = num;
                            break;
                        case 2:
                            a[0][j++] = num;
                            break;
                        case 3:
                            t[1][j++] = num;
                            break;
                        case 4:
                            a[1][j++] = num;
                            break;
                        default:
                            printf("第%d行default\n",line);
                            break;
                    }
                }
                flag = false;
            }
        }
        end = clock();
        printf("花费时间:%fs\t",(double)(end - start)/CLOCKS_PER_SEC);
        result = assembleLineSequence(e,t,a,x,n);
        printf("第%d组:%d\n",i,result);
        fclose(fp);
    }

    return 0;
}
