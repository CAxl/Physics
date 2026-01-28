
// GenerateEvent.cpp
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

class Particle {
public:
    Particle(double mIn, double pxIn, double pyIn, double pzIn) :
        m(mIn), px(pxIn), py(pyIn), pz(pzIn) {}

    inline double energy() const {
        return std::sqrt(m * m + px * px + py * py + pz * pz);
    }
private:
    double m, px, py, pz;

};

class Event {
public:
    Event(int n, int seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> U(-2.0, 2.0);
        const double m = 0.13957; // ~pion mass (GeV)
        parts.reserve(n);
        for (int i = 0; i < n; ++i)
            parts.push_back(Particle(m, U(rng), U(rng), U(rng)));
    }

    double totalEnergy() const {
        double s = 0.0;
        for (const auto& p : parts) s += p.energy();
        return s;
    }

private:
    std::vector<Particle> parts;

};

extern "C" {

    // Public C API: compute total energy of a generated Event
    double event_total_energy(int n, int seed) {
        Event ev(n, seed);
        return ev.totalEnergy();
    }

} // extern "C"
