#pragma once
#include <stdlib.h>

typedef int __boolean;

#define FALSE 0
#define TRUE 1

#define MAX(X, Y) (X > Y ? X : Y)

void* _mem_alloc(int size);
