# hector_index：Flat 稠密与稀疏索引、标量距离、索引工厂。只依赖 hector_common。
add_library(hector_index STATIC
  flat_index.cc
  flat_sparse_index.cc
  index_factory.cc
  metric.cc)
target_link_libraries(hector_index PUBLIC hector_common)
hector_target_defaults(hector_index)
