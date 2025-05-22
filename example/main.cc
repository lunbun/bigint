#include <iostream>

#include <bigint.h>

int main() {
    bigint_t a = 1;
    std::cout << "a = " << (a - 1).ToHexString() << std::endl;
}
