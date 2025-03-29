#include <type_traits>
#include <concepts>

template <typename T>
concept Real = requires(T t) {
    { t + t } -> std::same_as<T>;
    { t - t } -> std::same_as<T>;
    { t * t } -> std::same_as<T>;
    { t / t } -> std::same_as<T>;
    { -t } -> std::same_as<T>;
    { +t } -> std::same_as<T>;
    { t == t } -> std::same_as<bool>;
    { t != t } -> std::same_as<bool>;
    { t < t } -> std::same_as<bool>;
    { t <= t } -> std::same_as<bool>;
    { t > t } -> std::same_as<bool>;
    { t >= t } -> std::same_as<bool>;
    { t += t } -> std::same_as<T&>;
    { t -= t } -> std::same_as<T&>;
    { t *= t } -> std::same_as<T&>;
    { t /= t } -> std::same_as<T&>;
};