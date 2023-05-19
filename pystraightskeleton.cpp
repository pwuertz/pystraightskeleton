#include <algorithm>
#include <iostream>
#include <vector>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/create_straight_skeleton_2.h>
#include <CGAL/create_straight_skeleton_from_polygon_with_holes_2.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_2;
using Polygon = CGAL::Polygon_2<K>;
using PolygonWithHoles = CGAL::Polygon_with_holes_2<K>;
using Skeleton = CGAL::Straight_skeleton_2<K>;
using Vertex = Skeleton::Vertex;
using Halfedge = Skeleton::Halfedge;

namespace py = pybind11;

static constexpr std::string_view MODULE_VERSION_STR = PYBIND11_TOSTRING(MODULE_VERSION);

using VertexIdx = uint32_t;
using FloatType = double;
static_assert(sizeof(Point) == 2 * sizeof(FloatType));


template<typename DstT, typename SrcT>
std::span<DstT> view_array_as(py::array_t<SrcT>& a)
{
    constexpr auto NSrcPerDst = sizeof(DstT) / sizeof(SrcT);
    static_assert(sizeof(DstT) == NSrcPerDst * sizeof(SrcT));
    return { reinterpret_cast<DstT*>(a.mutable_data()), a.size() / NSrcPerDst };
}

template<typename DstT, typename SrcT>
std::span<const DstT> view_array_as(const py::array_t<SrcT>& a)
{
    constexpr auto NSrcPerDst = sizeof(DstT) / sizeof(SrcT);
    static_assert(sizeof(DstT) == NSrcPerDst * sizeof(SrcT));
    return { reinterpret_cast<const DstT*>(a.data()), a.size() / NSrcPerDst };
}

std::span<const Point> as_cgal_point_view(const py::array_t<FloatType>& array)
{
    // Require contiguous array of 2D points
    if (array.ndim() != 2 || array.shape(1) != 2) {
        throw std::runtime_error("Expected array of 2d points");
    }
    if (array.strides(0) != (2 * sizeof(FloatType)) || array.strides(1) != sizeof(FloatType)) {
        throw std::runtime_error("Expected contiguous array of points");
    }
    // Return view to array as CGAL 2D Point sequence
    return view_array_as<Point>(array);
}

Polygon as_cgal_polygon(const py::array_t<FloatType>& array)
{
    const auto points = as_cgal_point_view(array);
    return Polygon { points.begin(), points.end() };
}

std::vector<Polygon> as_cgal_polygons(const std::list<py::array_t<FloatType>>& array_list)
{
    std::vector<Polygon> out;
    out.reserve(array_list.size());
    for (const auto& item : array_list) {
        const auto array = py::cast<py::array_t<FloatType>>(item);
        out.emplace_back(as_cgal_polygon(array));
    }
    return out;
}

std::pair<const Vertex&, const Vertex&> get_edge_vertices(const Halfedge& e)
{
    return { *e.vertex(), *e.opposite()->vertex() };
}

template<typename T>
py::array_t<T> as_numpy(std::vector<std::pair<T, T>>&& vec)
{
    py::array_t<T> arr { ssize_t(2 * vec.size()) };
    auto arr_view = view_array_as<std::pair<T, T>>(arr);
    for (auto idx=0; auto val : vec) { arr_view[idx++] = val; }
    return arr.reshape({ ssize_t(vec.size()), ssize_t(2) });
}

template<typename T>
py::array_t<T> as_numpy(std::vector<T>&& vec)
{
    py::array_t<T> arr { ssize_t(vec.size()) };
    auto arr_view = view_array_as<T>(arr);
    for (auto idx=0; auto val : vec) { arr_view[idx++] = val; }
    return arr;
}

std::tuple<py::array_t<FloatType>, py::array_t<FloatType>, py::array_t<VertexIdx>>
  skeletonize(const py::array_t<FloatType>& outline, const std::optional<std::list<py::array_t<FloatType>>>& holes)
{
    // Create CGAL polygons from numpy coordinate arrays
    Polygon outline_polygon = as_cgal_polygon(outline);
    std::vector<Polygon> hole_polygons = (holes) ? as_cgal_polygons(*holes) : std::vector<Polygon>{};

    // Normalize orientation of outline & holes
    if (!outline_polygon.is_counterclockwise_oriented()) { outline_polygon.reverse_orientation(); }
    for (auto& hole_polygon : hole_polygons) {
        if (!hole_polygon.is_clockwise_oriented()) { hole_polygon.reverse_orientation(); }
    }

    // Create straight-skeleton from polygon
    // (no GIL during skeleton computation)
    const auto skeleton = [&]() {
        py::gil_scoped_release release;
        return CGAL::create_interior_straight_skeleton_2(
            outline_polygon.vertices_begin(),
            outline_polygon.vertices_end(),
            hole_polygons.begin(),
            hole_polygons.end()
        );
    }();

    // Count number of skeleton edges
    const auto is_skeleton_edge = [](const Halfedge& e) { return e.is_inner_bisector(); };
    const auto n_skeleton_edges = std::count_if(
        skeleton->halfedges_begin(), skeleton->halfedges_end(), is_skeleton_edge);

    // Iterate over skeleton edges, track vertex-id pairs and unique ids
    std::vector<std::pair<int, int>> vertex_id_pairs;
    std::unordered_set<int> vertex_ids;
    vertex_id_pairs.reserve(n_skeleton_edges);
    vertex_ids.reserve(n_skeleton_edges/2);
    std::for_each(skeleton->halfedges_begin(), skeleton->halfedges_end(), [&, idx=0](const Halfedge& e) mutable {
        if (is_skeleton_edge(e)) {
            const auto [v1, v2] = get_edge_vertices(e);
            vertex_id_pairs.emplace_back(v1.id(), v2.id());
            vertex_ids.emplace(v1.id());
            vertex_ids.emplace(v2.id());
        }
    });
    const auto n_skeleton_vertices = vertex_ids.size();

    // Re-enumerate skeleton vertices, gather coordinates and weights
    std::unordered_map<int, VertexIdx> vertex_idx_map;
    std::vector<std::pair<FloatType, FloatType>> vertex_coords;
    std::vector<FloatType> vertex_weights;
    vertex_idx_map.reserve(n_skeleton_vertices);
    vertex_coords.reserve(n_skeleton_vertices);
    vertex_weights.reserve(n_skeleton_vertices);
    std::for_each(skeleton->vertices_begin(), skeleton->vertices_end(), [&, idx=VertexIdx{0}](const Vertex& v) mutable {
        if (vertex_ids.contains(v.id())) {
            vertex_idx_map.emplace(v.id(), idx++);
            vertex_coords.emplace_back(v.point().x(), v.point().y());
            vertex_weights.push_back(v.time());
        }
    });

    // Transform vertex-id pairs to vertex-index pairs
    std::vector<std::pair<VertexIdx, VertexIdx>> vertex_idx_pairs;
    vertex_idx_map.reserve(n_skeleton_edges);
    for (const auto [id1, id2] : vertex_id_pairs) {
        vertex_idx_pairs.emplace_back(vertex_idx_map.at(id1), vertex_idx_map.at(id2));
    }
    std::sort(vertex_idx_pairs.begin(), vertex_idx_pairs.end());

    return {
        as_numpy(std::move(vertex_coords)),
        as_numpy(std::move(vertex_weights)),
        as_numpy(std::move(vertex_idx_pairs)),
    };
}

PYBIND11_MODULE(MODULE_NAME, m) {
    m.attr("__version__") = py::str(MODULE_VERSION_STR.data(), MODULE_VERSION_STR.size());

    m.def(
        "skeletonize", &skeletonize, "Calculate the straight skeleton of given polygon",
        py::arg("outline"), py::arg("holes") = py::none()
        );
}
