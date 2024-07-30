#pragma once
#include <stdlib.h>
#include <string.h>

typedef int __boolean;

#define FALSE 0
#define TRUE 1

#define MAX(X, Y) (X > Y ? X : Y)
#define MIN(X, Y) (X <= Y ? X : Y)

void* _mem_alloc(int size);
