#include "hector/common/types.h"

namespace hector {

bool HitLess(const Hit& a, const Hit& b) {
  if (a.distance != b.distance) {
    return a.distance < b.distance;
  }
  return a.row_id < b.row_id;
}

}  // namespace hector
