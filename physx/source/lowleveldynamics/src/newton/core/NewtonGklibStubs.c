/* METIS references GKlib's file removal from its on-disk graph mode, which the Newton
   ordering never enables. The trimmed GKlib omits fs.c; wasm-ld requires the symbol. */
int gk_rmpath(char* path)
{
	(void)path;
	return 0;
}
