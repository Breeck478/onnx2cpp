// dco/c++ version: v4.4.1
// branch: 4.4.1
// used DCO_FLAGS: -DDCO_SCRAMBLE -DDCO_LICENSE

// ================================================================ //
// *** This is a generated file from above given source version *** //
// ================================================================ //
#ifndef DCO_HPP
#define DCO_HPP

#include "dco_interoperability_eigen.hpp"

#include "dco_configuration.hpp"
#include "dco_global_includes.hpp"
#include "dco_logging.hpp"

//** Unfortunately, old intel compilers emit a warning in various
//** functions using 'if constexpr' due to missing return
//** statement. That's a false positive, therefore we suppress
//** it. This needs to be done globally (not popped after including
//** dco.hpp), since the warning is actually emitted in the
//** instantiations of the functions, i.e., in user's code.
#if defined(__INTEL_COMPILER) && __INTEL_COMPILER_BUILD_DATE < 20220000
#pragma warning( disable: 1011 )
#endif

#ifdef DCO_NO_INTERMEDIATES
#include "dco_inc_noet.hpp"
#else
#include "dco_inc.hpp"
#endif

#include "dco_vector.hpp"

#include "dco_std_compatibility.hpp"
#include "dco_allocation_helper.hpp"

#include "dco_interoperability_eigen_spec.hpp"
#include "dco_interoperability_boost_interval.hpp"

#ifdef DCO_GOLD
# include "dco_codegen_helper.hpp"
#endif

#endif  
