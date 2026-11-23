#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the Kitsune project.  It is released under the LLVM
# license.
#

include(GetClangResourceDir)

get_clang_resource_dir(CLANG_RESOURCE_INCDIR
  PREFIX ${LLVM_LIBRARY_OUTPUT_INTDIR}/..
  SUBDIR include
)

set(out_files)
function(copy_header_to_resource_dir src file)
  set(dst "${CLANG_RESOURCE_INCDIR}/${file}")
  add_custom_command(OUTPUT "${dst}"
    DEPENDS "${src}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}" "${dst}"
    COMMENT "Copying ${file}..."
    VERBATIM)
  list(APPEND out_files "${dst}")
  set(out_files "${out_files}" PARENT_SCOPE)
endfunction(copy_header_to_resource_dir)

copy_header_to_resource_dir("${CMAKE_CURRENT_SOURCE_DIR}/kitsune.h" kitsune.h)

if(KITSUNE_OPENCILK_ENABLED)
  set(cilkscale_header "${KITSUNE_INCLUDE_DIR}/cilk/cilkscale.h")
  copy_header_to_resource_dir("${cilkscale_header}" cilk/cilkscale.h)
  install(FILES "${cilkscale_header}"
    DESTINATION "${CLANG_RESOURCE_SUBDIR}/include/cilk"
    COMPONENT "kitsune-resource-headers")
endif()

add_custom_target("kitsune-resource-headers" ALL DEPENDS ${out_files})

# Kitsune is registered as a project and will show up as a "tool" alongside
# clang. But we actually want this configuration file in a place where clang
# will pick it up.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config/config.h.cmake
  ${CMAKE_BINARY_DIR}/include/kitsune/Config/config.h)

install(DIRECTORY
  ${CMAKE_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES kitsune.h
  DESTINATION "${CLANG_RESOURCE_SUBDIR}/include"
  COMPONENT "kitsune-resource-headers")
