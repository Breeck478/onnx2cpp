#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
int main(int argc, char *argv[])
{
	
	if (argc < 2) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << "./TestMakeFileGenerator <fileName> <directory> " << std::endl;
		std::cerr << std::endl;
		std::cerr << " <fileName> is the name of the generate file" << std::endl;
		std::cerr << " <directory> is the directory of the include files for dco" << std::endl;
		exit(1);
	}
	std::string name(argv[1]);
	std::string dir(argv[2]);
	std::string targetDir(argv[3]);
	std::fstream file;
	try {
		file.open(targetDir, std::ostringstream::out | std::ostringstream::trunc);
	}
	catch (const std::exception& e) {
		std::cerr << "Error opening target file: " << e.what() << std::endl;
		exit(1);
	}
	file << "DCO_INC_DIR=$(shell wslpath \"" << dir<< "include/\")\n";
	file << "LOCAL_XTENSOR_INC=$(shell wslpath \"" << dir <<"\")\n";
	file << "LOCAL_OP_DIR=$(shell wslpath \"" << dir << "../\")\n";
	file << "DCO_LIB_DIR=$(shell wslpath \""<< dir << "lib/\")\n" ;
	file << "DCO_FLAGS=-DDCO_DISABLE_AUTO_WARNING -DDCO_DISABLE_AVX2_WARNING -DDCO_EXT_EIGEN_IGNORE_VERSION -DDCO_CHUNK_TAPE -DDCO_NO_INTERMEDIATES\n";// -DCO_DEBUG\n" ;
	file << "CPPC=g++ -Wall -std=c++20\n";
	file << "\n";
	file << "all : "<< name << ".exe\n";
	file << "\t./" << name << ".exe\n";

	file << "%.exe : %.cpp\n";
	file << "	$(CPPC) -I$(DCO_INC_DIR) -I$(LOCAL_XTENSOR_INC) -I$(LOCAL_OP_DIR)  $(DCO_FLAGS) -O3 $< -o $@ -L$(DCO_LIB_DIR) -ldcoc\n";

	file << "clean :\n";
	file << "\trm -f "<< name <<".exe\n";

	file << ".PHONY: all clean\n";
	file.close();
}
