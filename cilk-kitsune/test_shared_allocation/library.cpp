unsigned char *allocate_buffer() { return new unsigned char[64]; }

void release_buffer(unsigned char *buffer) { delete[] buffer; }
