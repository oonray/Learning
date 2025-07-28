#ifndef __sockets_unix_h
#define __sockets_unix_h

#include <iostream>
#include <sys/socket.h>
#include <sys/select.h>
#include <errno.h>

namespace lrn {

typedef struct sock {
    std::streambuf *in;
} sock;

};
#endif
