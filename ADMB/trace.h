#ifndef __TRACE__
#define __TRACE__

/* The log file.
#include <fstream>
using std::ofstream;
Must be declared and opened in a different file, usually the file containing
main()
*/

#include <fstream>
using std::ofstream;
extern ofstream Clogf;

#undef STREAM
//#define STREAM std::cout
#define STREAM Clogf

#undef HERE
/**
\def HERE
Indicates the line number of the file which the statement appears.
*/
#define HERE STREAM  << "reached " << __LINE__ << " in " << __FILE__ << "\n";

#undef HALT
/**
\def HALT
Prints the file name and line number and exits with exict code = 1.
*/
#define HALT STREAM <<"\nBailing out in file"<<__FILE__<<" at line " <<__LINE__<< std::endl; exit(1);

#undef TRACE
/**
\def TRACE
Prints the value of its argument, file name and line number.
*/
#define TRACE(object) STREAM << "line " << __LINE__ << ", file " << __FILE__ << ", " << #object " = " << object << "\n";

#undef TTRACE
/**
\def TTRACE
Prints the value of two arguments (note the double 'T'), file name and line number.
*/
#define TTRACE(o1,o2) STREAM << "line " << __LINE__ << ", file " << __FILE__ << ", " << #o1 " = " << o1<< ", " << #o2 " = " << o2 << "\n";

#undef ASSERT
/** 
\def ASSERT
It the argument is logically false, prints the file name, line number and value of argument and exits with exict code = 1.
*/
#define ASSERT(object) if (!object) { STREAM << "ASSERT: line = " << __LINE__ << " file = " << __FILE__ << " " << #object << " = " << object << " (false)\n"; exit(1); }
#endif

#ifndef NLL_TRACE
#define NLL_TRACE(nll) nll_vector(nll_count++) = value(nll);
#endif
