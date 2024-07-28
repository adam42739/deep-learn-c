#include "stdlib_mod.h"

void* _mem_alloc(int size)
{
	void* ptr = malloc(size);
	if (!ptr)
	{
		exit(-1);
	}
	return ptr;
}
