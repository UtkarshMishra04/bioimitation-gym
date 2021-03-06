/**
 * @file inverse_dynamics.cpp
 *
 * \brief An interface for getting access to various useful inverse dynamics
 * operations.
 *
 * @author Dimitar Stanev <dimitar.stanev@epfl.ch>
 */
#include <OpenSim/Simulation/Model/Model.h>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

using OpenSim::Model;
using SimTK::SpatialVec;
using SimTK::Stage;
using SimTK::State;
using SimTK::Vector;
using SimTK::Vector_;
using std::string;
using namespace boost::python;

template< typename T > inline
std::vector< T > to_std_vector( const object& iterable )
{
    return std::vector<T>(stl_input_iterator< T >( iterable ),
                          stl_input_iterator< T >( ) );
}

template <class T> inline Vector_<T> py_list_to_simtk_vector(const list& l) {
    Vector_<T> vec(len(l));
    for (int i = 0; i < len(l); ++i)
        vec[i] = extract<T>(l[i]);
    return vec;
}

template <class T> inline list simtk_vector_to_py_list(const Vector_<T>& vec) {
    list l;
    //for (auto iter = vec.begin(); iter != vec.end(); ++iter)
    for (int i = 0; i < vec.size(); i++)
        l.append(vec[i]);
    return l;
}

class InverseDynamics {
public:
    // Muscle forces are disabled internally
    InverseDynamics(const string& modelFile)
        : model(modelFile) {
        // disable muscles, otherwise they apply passive forces
        state = model.initSystem();
        for (int i = 0; i < model.getMuscles().getSize(); ++i)
            model.updForceSet()[i].setAppliesForce(state, false);
    }

    void setStateAndRealizeDynamics(const double& t, const list& q,
                                    const list& qDot) {
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        state.setU(py_list_to_simtk_vector<double>(qDot));
        model.getMultibodySystem().realize(state, Stage::Dynamics);
    }

    // Calculate residual forces (inverse dynamics) such that tau_residual = M *
    // qDDot_v + f_internal - f_applied
    list calculateResidualForces(const double& t, const list& q,
                                 const list& qDot, const list& qDDot) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        state.setU(py_list_to_simtk_vector<double>(qDot));
        auto qDDot_v = py_list_to_simtk_vector<double>(qDDot);
        // state.updUDot() = qDDot_v;

        // realize to dynamics stage so that all model forces are computed
        model.getMultibodySystem().realize(state, Stage::Dynamics);

        // get applied mobility (generalized) forces generated by
        // components of the model, like actuators
        const Vector& appliedMobilityForces = model.getMultibodySystem()
            .getMobilityForces(state, Stage::Dynamics);

        // get all applied body forces like those from contact
        const Vector_<SpatialVec>& appliedBodyForces = model.getMultibodySystem()
            .getRigidBodyForces(state, Stage::Dynamics);

        // tau_residual = M * qDDot_v + f_internal - f_applied
        Vector tau_residual;
        model.getMultibodySystem().getMatterSubsystem()
            .calcResidualForceIgnoringConstraints(state,
                                                  appliedMobilityForces,
                                                  appliedBodyForces,
                                                  qDDot_v,
                                                  tau_residual);
        return simtk_vector_to_py_list<double>(tau_residual);
    }

    // Calculate total forces f such that M qddot + f = tau
    list calculateTotalForces(const double& t, const list& q, const list& qDot) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        state.setU(py_list_to_simtk_vector<double>(qDot));

        // realize to dynamics stage so that all model forces are computed
        model.getMultibodySystem().realize(state, Stage::Dynamics);

        // get acting torques and body forces
        auto bodyForces = model.getMultibodySystem()
            .getRigidBodyForces(state, Stage::Dynamics);
        auto generalizedForces = model.getMultibodySystem()
            .getMobilityForces(state, Stage::Dynamics);

        // map body forces to joint forces
        Vector jointForces;
        model.getMatterSubsystem()
            .multiplyBySystemJacobianTranspose(state, bodyForces, jointForces);

        // calculate Coriolis
        Vector c;
        model.getMatterSubsystem().calcResidualForceIgnoringConstraints(
            state, Vector(0), Vector_<SpatialVec>(0), Vector(0), c);

        // calculate total forces f such that M qddot + f = tau
        auto f = c - generalizedForces - jointForces;

        return simtk_vector_to_py_list<double>(f);
    }

    // Calculate gravity such that M qddot + c = g + tau
    list calculateGravity(const double& t, const list& q) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));

        // realize to Position
        model.getMultibodySystem().realize(state, Stage::Position);

        // calculate gravity (M qddot + c = g + tau)
        Vector g;
        model.getMatterSubsystem().multiplyBySystemJacobianTranspose(
            state, model.getGravityForce().getBodyForces(state), g);

        return simtk_vector_to_py_list<double>(g);
    }

    // Calculate Coriolis c such that M qddot + c = g + tau
    list calculateCoriolis(const double& t, const list& q,
                           const list& qDot) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        state.setU(py_list_to_simtk_vector<double>(qDot));

        // realize to dynamics stage so that all model forces are computed
        model.getMultibodySystem().realize(state, Stage::Velocity);

        // calculate Coriolis (M qddot + c = g + tau)
        Vector c;
        model.getMatterSubsystem().calcResidualForceIgnoringConstraints(
            state, Vector(0), Vector_<SpatialVec>(0), Vector(0), c);

        return simtk_vector_to_py_list<double>(c);
    }

    // Calculate the product M * a efficiently
    list multiplyByM(const double& t, const list& q, const list& a) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        Vector a_v = py_list_to_simtk_vector<double>(a);

        model.getMultibodySystem().realize(state, Stage::Position);

        // calculate the product of M * a efficiently
        Vector Ma;
        model.getMultibodySystem().getMatterSubsystem()
            .multiplyByM(state, a_v, Ma);

        return simtk_vector_to_py_list<double>(Ma);
    }

    // Calculate the product M^-1 * tau efficiently
    list multiplyByMInv(const double& t, const list& q, const list& tau) {
        // update state
        state.setTime(t);
        state.setQ(py_list_to_simtk_vector<double>(q));
        Vector tau_v = py_list_to_simtk_vector<double>(tau);

        model.getMultibodySystem().realize(state, Stage::Position);

        // calculate the product of M^-1 * tau efficiently
        Vector MInv_tau;
        model.getMultibodySystem().getMatterSubsystem()
            .multiplyByMInv(state, tau_v, MInv_tau);

        return simtk_vector_to_py_list<double>(MInv_tau);
    }

protected:
    Model model;
    State state;
};

BOOST_PYTHON_MODULE(inverse_dynamics)
{
  class_<InverseDynamics>("InverseDynamics", init<string>())
      .def("setStateAndRealizeDynamics", &InverseDynamics::setStateAndRealizeDynamics)
      .def("calculateResidualForces", &InverseDynamics::calculateResidualForces)
      .def("calculateTotalForces", &InverseDynamics::calculateTotalForces)
      .def("calculateGravity", &InverseDynamics::calculateGravity)
      .def("calculateCoriolis", &InverseDynamics::calculateCoriolis)
      .def("multiplyByM", &InverseDynamics::multiplyByM)
      .def("multiplyByMInv", &InverseDynamics::multiplyByMInv)
    ;
}
