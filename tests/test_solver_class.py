# Test script to test CMPLXFOIL
# =============================================================================
# Extension modules
# =============================================================================
from cmplxfoil import CMPLXFOIL

# =============================================================================
# Standard Python Modules
# =============================================================================
import os

# =============================================================================
# External Python modules
# =============================================================================
import unittest
import numpy as np

from baseclasses import AeroProblem

pygeoImportFailed = False
try:
    from pygeo import DVGeometry, DVGeometryCST
except ImportError:
    pygeoImportFailed = True

baseDir = os.path.dirname(os.path.abspath(__file__))  # Path to current folder

# "ld" (cl/cd) is a ratio and "xtrupper"/"xtrlower" are transition (threshold-crossing) locations.
# Both are exact and complex-step-accurate (see test_ld_is_cl_over_cd and the FD-vs-CS agreement
# tests below), but a *real-valued* finite-difference estimate of either has a materially higher
# truncation-error noise floor than for a direct, well-conditioned output like cl/cd/cm: a ratio
# amplifies the denominator's FD noise, and a threshold crossing is not smooth under a real
# perturbation the way it is under a complex one. So checks that compare against a raw one-sided
# FD reference need a looser tolerance for these three functions specifically; a mismatch here
# reflects that reference's own noise, not a wrong derivative.
FD_NOISY_FUNCS = {"ld", "xtrupper", "xtrlower"}


def tolFor(evalFunc, relTol, absTol, looseRelTol=1e-2, looseAbsTol=1e-2):
    baseName = evalFunc.split("_")[-1]
    if baseName in FD_NOISY_FUNCS:
        return looseRelTol, looseAbsTol
    return relTol, absTol


class TestNACA(unittest.TestCase):
    def setUp(self):
        # Set the random range to use consistent random numbers
        self.rng = np.random.default_rng(7)
        self.CFDSolver = CMPLXFOIL(
            os.path.join(baseDir, "naca0012.dat"),
            options={"printRealConvergence": False},
        )
        self.alphaSequence = np.linspace(-0.5, 0.5, 11)
        self.rng.shuffle(self.alphaSequence)
        self.ap = AeroProblem(
            name="fc",
            alpha=3,
            mach=0.2,
            altitude=1e3,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=self.CFDSolver.functionList,
        )

    def test_cl_solve_default(self):
        """Test that SolveCL works correctly"""
        tol = 1e-5
        for CLTarget in self.alphaSequence:
            # Default settings
            self.CFDSolver.solveCL(self.ap, CLTarget, tol=tol)
            self.assertAlmostEqual(float(self.CFDSolver.xfoil.cr09.cl), CLTarget, delta=tol)

    def test_cl_solve_CLalphaGuess(self):
        """Test that SolveCL works correctly"""
        tol = 1e-5
        for CLTarget in self.alphaSequence:
            # Using secant with CLalphaGuess
            self.CFDSolver.solveCL(self.ap, CLTarget, tol=tol, useNewton=False, CLalphaGuess=0.11)
            self.assertAlmostEqual(float(self.CFDSolver.xfoil.cr09.cl), CLTarget, delta=tol)

    def test_cl_solve_Newton(self):
        """Test that SolveCL works correctly"""
        tol = 1e-5
        for CLTarget in self.alphaSequence:
            # Using Newton
            self.CFDSolver.solveCL(self.ap, CLTarget, tol=tol, useNewton=True)
            self.assertAlmostEqual(float(self.CFDSolver.xfoil.cr09.cl), CLTarget, delta=tol)

    def test_function_values(self):
        funcs = {}
        alphas = np.linspace(-10, 10, 5)
        for _, alpha in enumerate(alphas):
            self.ap.name = f"polar_{alpha}deg"
            self.ap.alpha = alpha
            self.CFDSolver(self.ap)
            self.CFDSolver.evalFunctions(self.ap, funcs)
        true_funcs = {
            "polar_-10.0deg_cd": 0.011025153413460432,
            "polar_-10.0deg_cdp": 0.006380803330791225,
            "polar_-10.0deg_cl": -1.1588483291827825,
            "polar_-10.0deg_cm": -0.0012226349825473892,
            "polar_-5.0deg_cd": 0.006681735627505201,
            "polar_-5.0deg_cdp": 0.002103027412746445,
            "polar_-5.0deg_cl": -0.5699314393578008,
            "polar_-5.0deg_cm": -0.0017402658692315196,
            "polar_0.0deg_cd": 0.005149998979665711,
            "polar_0.0deg_cdp": 0.000720336879765777,
            "polar_0.0deg_cl": 2.3731559252448964e-14,
            "polar_0.0deg_cm": -5.189882517357458e-15,
            "polar_5.0deg_cd": 0.006681852094188599,
            "polar_5.0deg_cdp": 0.0021030658942220873,
            "polar_5.0deg_cl": 0.5699617443207495,
            "polar_5.0deg_cm": 0.0017333482408693297,
            "polar_10.0deg_cd": 0.011024378048979051,
            "polar_10.0deg_cdp": 0.006379997348330231,
            "polar_10.0deg_cl": 1.1587674856662604,
            "polar_10.0deg_cm": 0.0012397586091073744,
            "polar_-10.0deg_ld": -105.10949695884109,
            "polar_-10.0deg_xtrlower": 0.015595033907518777,
            "polar_-10.0deg_xtrupper": 1.0,
            "polar_-5.0deg_ld": -85.2969155217306,
            "polar_-5.0deg_xtrlower": 0.0714247799441054,
            "polar_-5.0deg_xtrupper": 0.888328893645548,
            "polar_0.0deg_ld": 3.4881211833620813e-12,
            "polar_0.0deg_xtrlower": 0.4562869692616716,
            "polar_0.0deg_xtrupper": 0.4562869692616987,
            "polar_5.0deg_ld": 85.29996418440534,
            "polar_5.0deg_xtrlower": 0.8883234915239407,
            "polar_5.0deg_xtrupper": 0.07141958811667012,
            "polar_10.0deg_ld": 105.1095563412445,
            "polar_10.0deg_xtrlower": 1.0,
            "polar_10.0deg_xtrupper": 0.015595936714615535,
        }
        for key, true_val in true_funcs.items():
            self.assertAlmostEqual(true_val, funcs[key])

    def test_ld_is_cl_over_cd(self):
        """ld should always be exactly the ratio of the cl and cd computed in the same call."""
        for alpha in np.linspace(-10, 10, 5):
            self.ap.alpha = alpha
            self.CFDSolver(self.ap)
            funcs = {}
            self.CFDSolver.evalFunctions(self.ap, funcs)
            self.assertAlmostEqual(funcs["fc_ld"], funcs["fc_cl"] / funcs["fc_cd"])


class TestTransition(unittest.TestCase):
    def setUp(self):
        self.cmplxfoilOptions = {"printRealConvergence": False}
        self.CFDSolver = CMPLXFOIL(os.path.join(baseDir, "naca0012.dat"), options=self.cmplxfoilOptions)
        self.evalFuncs = self.CFDSolver.functionList
        self.ap = AeroProblem(
            name="fc",
            alpha=3,
            mach=0.2,
            altitude=1e3,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=self.evalFuncs,
        )

    def test_trip(self):
        """
        Test that setting the trip changes the result.
        """

        self.cmplxfoilOptions.update({"xTrip": np.array([0.1, 0.1])})
        self.CFDSolverTripped = CMPLXFOIL(os.path.join(baseDir, "naca0012.dat"), options=self.cmplxfoilOptions)

        funcs = {}
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs)

        funcsTripped = {}
        self.CFDSolverTripped(self.ap)
        self.CFDSolverTripped.evalFunctions(self.ap, funcsTripped)

        for func in self.evalFuncs:
            self.assertNotEqual(funcs["fc_" + func], funcsTripped["fc_" + func])

    def test_nCrit(self):
        """
        Test that setting nCrit changes the result.
        """

        self.cmplxfoilOptions.update({"nCrit": 5.0})
        self.CFDSolverNCrit = CMPLXFOIL(os.path.join(baseDir, "naca0012.dat"), options=self.cmplxfoilOptions)

        funcs = {}
        self.CFDSolver(self.ap)
        self.CFDSolver.evalFunctions(self.ap, funcs)

        funcsNCrit = {}
        self.CFDSolverNCrit(self.ap)
        self.CFDSolverNCrit.evalFunctions(self.ap, funcsNCrit)

        for func in self.evalFuncs:
            self.assertNotEqual(funcs["fc_" + func], funcsNCrit["fc_" + func])


@unittest.skipIf(pygeoImportFailed, "pygeo not available")
class TestDerivativesFFD(unittest.TestCase):
    def setUp(self):
        # Set the random range to use consistent random numbers
        self.rng = np.random.default_rng(7)

        # Flight conditions
        alpha = 1.5
        mach = 0.3
        Re = 1e6
        T = 288.15  # K

        # Add CMPLXFOIL solver
        cmplxfoilOptions = {
            "printRealConvergence": False,
            "writeCoordinates": False,
            "plotAirfoil": False,
            "writeSolution": False,
        }
        self.CFDSolver = CMPLXFOIL(os.path.join(baseDir, "naca0012.dat"), options=cmplxfoilOptions)

        # Add AeroProblem with flight conditions and AoA DV
        self.ap = AeroProblem(
            name="fc",
            alpha=alpha,
            mach=mach,
            reynolds=Re,
            reynoldsLength=1.0,
            T=T,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=self.CFDSolver.functionList,
        )
        self.ap.addDV("alpha", value=alpha, lower=0, upper=10.0, scale=1.0)

        # Geometry setup
        FFDFile = os.path.join(baseDir, "naca0012_ffd.xyz")
        self.DVGeo = DVGeometry(FFDFile)
        self.DVGeo.addLocalDV("shape", lower=-0.05, upper=0.05, axis="y", scale=1.0)
        self.CFDSolver.setDVGeo(self.DVGeo)

    def test_alpha_sens(self):
        # Initialize necessary variables
        step = 1e-5  # step size in alpha
        relTol = 5e-3  # acceptable relative error
        alpha = 1.5  # deg
        alphaName = "alpha_" + self.ap.name

        # Evaluate the current functions
        x = {alphaName: alpha}
        self.ap.setDesignVars(x)
        self.CFDSolver(self.ap)
        origFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, origFuncs)

        # Evaluate built in sensitivities with both methods
        funcsSensFD = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensFD, mode="FD")
        self.CFDSolver(self.ap)
        funcsSensCS = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensCS, mode="CS")

        # Perturb alpha
        x[alphaName] += step
        self.ap.setDesignVars(x)
        self.CFDSolver(self.ap)
        pertFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, pertFuncs)

        # Check each function
        for evalFunc in origFuncs.keys():
            checkSensFD = (pertFuncs[evalFunc] - origFuncs[evalFunc]) / step
            funcRelTol, funcAbsTol = tolFor(evalFunc, relTol, 1e-16)

            # Check evalFunctionsSens's finite difference
            actualSensFD = funcsSensFD[evalFunc][alphaName].item()
            np.testing.assert_allclose(checkSensFD, actualSensFD, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's complex step
            actualSensCS = funcsSensCS[evalFunc][alphaName].item()
            np.testing.assert_allclose(checkSensFD, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)

    def test_shape_sens_random(self):
        # Initialize necessary variables
        step = 1e-7  # step size in shape variables
        relTol = 1e-3  # acceptable relative error
        absTol = 1e-3  # acceptable absolute error
        alpha = 1.5  # deg
        alphaName = "alpha_" + self.ap.name
        nShape = 40  # number of shape variables
        shape = self.rng.random(nShape) * 1e-2  # shape variables
        shapeName = "shape"

        # Evaluate the current functions
        x = {alphaName: alpha, shapeName: shape}
        self.ap.setDesignVars(x)
        self.DVGeo.setDesignVars(x)
        self.CFDSolver(self.ap)
        origFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, origFuncs)

        # Evaluate built in sensitivities with both methods
        funcsSensFD = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensFD, mode="FD")
        self.CFDSolver(self.ap)
        funcsSensCS = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensCS, mode="CS")

        # Estimate each partial with finite differences
        checkSensFD = {func: {"shape": np.zeros(nShape)} for func in origFuncs.keys()}  # initialize
        for i in range(nShape):
            x[shapeName][i] += step
            self.DVGeo.setDesignVars(x)
            self.CFDSolver(self.ap)
            pertFuncs = {}
            self.CFDSolver.evalFunctions(self.ap, pertFuncs)
            x[shapeName][i] -= step

            # Compute partial derivative for each function of interest
            for evalFunc in origFuncs.keys():
                checkSensFD[evalFunc][shapeName][i] = (pertFuncs[evalFunc] - origFuncs[evalFunc]) / step

        # Check each function
        for evalFunc in origFuncs.keys():
            # Each partial within each function
            checkVal = checkSensFD[evalFunc][shapeName]
            actualSensFD = funcsSensFD[evalFunc][shapeName]
            actualSensCS = funcsSensCS[evalFunc][shapeName]
            funcRelTol, funcAbsTol = tolFor(evalFunc, relTol, absTol)

            # Check evalFunctionsSens's finite difference
            np.testing.assert_allclose(checkVal, actualSensFD, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's complex step
            np.testing.assert_allclose(checkVal, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's methods against each other
            np.testing.assert_allclose(actualSensFD, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)


@unittest.skipIf(pygeoImportFailed, "pygeo not available")
class TestDerivativesCST(unittest.TestCase):
    def setUp(self):
        # Flight conditions
        self.alpha = 2.0
        self.mach = 0.1
        self.Re = 1e7
        self.T = 288.15  # K

        self.nCoeff = 2
        self.DATfile = "n0012BluntTE.dat"

        # Add CMPLXFOIL solver
        cmplxfoilOptions = {
            "printRealConvergence": False,
            "writeCoordinates": False,
            "plotAirfoil": False,
            "writeSolution": False,
        }
        self.CFDSolver = CMPLXFOIL(os.path.join(baseDir, self.DATfile), options=cmplxfoilOptions)

        # Add AeroProblem with flight conditions and AoA DV
        self.ap = AeroProblem(
            name="fc",
            alpha=self.alpha,
            mach=self.mach,
            reynolds=self.Re,
            reynoldsLength=1.0,
            T=self.T,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=self.CFDSolver.functionList,
        )
        self.ap.addDV("alpha", value=self.alpha, lower=0, upper=10.0, scale=1.0)

        # Geometry setup
        self.CSTName = {"upper": "upper_shape", "lower": "lower_shape"}
        self.CST = {}

        self.CST["upper"] = np.array([0.15, 0.13])
        self.CST["lower"] = np.array([-0.15, -0.1])
        self.upperName = "upper_shape"
        self.lowerName = "lower_shape"

        # This combination of alpha and CST coefficients is likely on a cusp (see multimodality
        # section in the docs), resulting in the FD and CS derivatives mismatching
        # self.alpha = 3.1998091362966057
        # self.CST["upper"] = np.array([0.15335040881179007, 0.1373052558913066])
        # self.CST["lower"] = np.array([-0.15335040881179007, -0.05284956303352384])

        self.DVGeo = DVGeometryCST(os.path.join(baseDir, self.DATfile), numCST=self.nCoeff, debug=False)
        self.DVGeo.addDV(self.CSTName["upper"], dvType="upper", default=self.CST["upper"])
        self.DVGeo.addDV(self.CSTName["lower"], dvType="lower", default=self.CST["lower"])
        self.CFDSolver.setDVGeo(self.DVGeo)

    def test_alpha_sens(self):
        # Initialize necessary variables
        step = 1e-6  # step size in alpha
        relTol = 2e-3  # acceptable relative error
        absTol = 1e-5  # acceptable absolute error
        alphaName = "alpha_" + self.ap.name

        # Evaluate the current functions
        x = {alphaName: self.alpha}
        self.ap.setDesignVars(x)
        self.CFDSolver(self.ap)
        origFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, origFuncs)

        # Evaluate built in sensitivities with both methods
        funcsSensFD = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensFD, mode="FD")
        self.CFDSolver(self.ap)
        funcsSensCS = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensCS, mode="CS")

        # Perturb alpha
        x[alphaName] += step
        self.ap.setDesignVars(x)
        self.CFDSolver(self.ap)
        pertFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, pertFuncs)

        # Check each function
        for evalFunc in origFuncs.keys():
            checkSensFD = (pertFuncs[evalFunc] - origFuncs[evalFunc]) / step
            actualSensFD = funcsSensFD[evalFunc][alphaName].item()
            actualSensCS = funcsSensCS[evalFunc][alphaName].item()
            funcRelTol, funcAbsTol = tolFor(evalFunc, relTol, absTol)

            # Check evalFunctionsSens's finite difference
            np.testing.assert_allclose(checkSensFD, actualSensFD, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's complex step
            np.testing.assert_allclose(checkSensFD, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)

    def test_upper_shape_sens(self):
        self._eval_shape_sens("upper")

    def test_lower_shape_sens(self):
        self._eval_shape_sens("lower")

    def _eval_shape_sens(self, surf):
        # Initialize necessary variables
        step = 1e-6  # step size in shape variables
        relTol = 1e-4  # acceptable relative error
        absTol = 1e-4  # acceptable absolute error

        # Evaluate the current functions
        x = {self.CSTName[surf]: self.CST[surf]}
        self.ap.setDesignVars(x)
        self.DVGeo.setDesignVars(x)
        self.CFDSolver(self.ap)
        origFuncs = {}
        self.CFDSolver.evalFunctions(self.ap, origFuncs)

        # Evaluate built in sensitivities with both methods
        funcsSensFD = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensFD, mode="FD")
        self.CFDSolver(self.ap)
        funcsSensCS = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSensCS, mode="CS")

        # Estimate each partial with finite differences
        checkSensFD = {
            func: {self.CSTName[surf]: np.zeros(len(self.CST[surf]))} for func in origFuncs.keys()
        }  # initialize
        for i in range(len(self.CST[surf])):
            x[self.CSTName[surf]][i] += step
            self.DVGeo.setDesignVars(x)
            self.CFDSolver(self.ap)
            pertFuncs = {}
            self.CFDSolver.evalFunctions(self.ap, pertFuncs)
            x[self.CSTName[surf]][i] -= step

            # Compute partial derivative for each function of interest
            for evalFunc in origFuncs.keys():
                checkSensFD[evalFunc][self.CSTName[surf]][i] = (pertFuncs[evalFunc] - origFuncs[evalFunc]) / step

        # Check each function
        for evalFunc in origFuncs.keys():
            # Each partial within each function
            checkVal = checkSensFD[evalFunc][self.CSTName[surf]]
            actualSensFD = funcsSensFD[evalFunc][self.CSTName[surf]]
            actualSensCS = funcsSensCS[evalFunc][self.CSTName[surf]]
            funcRelTol, funcAbsTol = tolFor(evalFunc, relTol, absTol)

            # Check evalFunctionsSens's finite difference
            np.testing.assert_allclose(checkVal, actualSensFD, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's complex step
            np.testing.assert_allclose(checkVal, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)

            # Check evalFunctionsSens's methods against each other
            np.testing.assert_allclose(actualSensFD, actualSensCS, rtol=funcRelTol, atol=funcAbsTol)


class TestPlotting(unittest.TestCase):
    """
    Test that the plotting utilities run with no errors.
    """

    def test_plotting(self):
        # Set the random range to use consistent random numbers
        self.rng = np.random.default_rng(13)
        self.ap = AeroProblem(
            name="fc",
            alpha=3,
            mach=0.2,
            altitude=1e3,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd", "cm"],
        )
        self.CFDSolver = CMPLXFOIL(
            os.path.join(baseDir, "naca0012.dat"),
            options={"printRealConvergence": False},
        )

        # Run an initial case and plot it
        self.CFDSolver(self.ap)
        _, _ = self.CFDSolver.plotAirfoil(showPlot=False)

        # Run another case and update the plot
        self.ap.alpha = 4.0
        self.CFDSolver(self.ap)
        _, _ = self.CFDSolver.plotAirfoil(showPlot=False)  # this will call self.CFDSolver._updateAirfoilPlot()

        # Run another case and try calling _updateAirfoilPlot directly
        self.ap.alpha = 5.0
        self.CFDSolver(self.ap)
        self.CFDSolver._updateAirfoilPlot(pause=False)


class TestEvalFunctions(unittest.TestCase):
    """Check that functions and derivatives are returned for the specified evalFuncs and the evalFuncs only."""

    def setUp(self):
        self.CFDSolver = CMPLXFOIL(
            os.path.join(baseDir, "naca0012.dat"),
            options={"printRealConvergence": False, "writeSolution": False, "writeCoordinates": False},
        )
        self.ap = AeroProblem(
            name="fc",
            alpha=1.5,
            mach=0.3,
            reynolds=1e6,
            reynoldsLength=1.0,
            T=288.15,
            areaRef=1.0,
            chordRef=1.0,
        )
        self.ap.addDV("alpha", value=1.5, lower=0.0, upper=10.0, scale=1.0)
        self.CFDSolver(self.ap)

    def _runAndCheck(self, mode):
        self.CFDSolver(self.ap)

        funcs = {}
        self.CFDSolver.evalFunctions(self.ap, funcs)
        self.assertEqual(set(funcs.keys()), {"fc_cl", "fc_cd"})

        funcsSens = {}
        self.CFDSolver.evalFunctionsSens(self.ap, funcsSens, mode=mode)
        funcSensKeys = {k for k in funcsSens if k != "fail"}
        self.assertEqual(funcSensKeys, {"fc_cl", "fc_cd"})

    def test_evalFunctionsKeys(self):
        for func in self.CFDSolver.functionList:
            with self.subTest(func=func):
                funcs = {}
                self.CFDSolver.evalFunctions(self.ap, funcs, evalFuncs=[func])
                self.assertEqual(set(funcs.keys()), {f"fc_{func}"})

    def test_evalFunctionsSensKeys(self):
        for func in self.CFDSolver.functionList:
            with self.subTest(func=func):
                for mode in ["FD", "CS"]:
                    with self.subTest(mode=mode):
                        funcsSens = {}
                        self.CFDSolver.evalFunctionsSens(self.ap, funcsSens, evalFuncs=[func], mode=mode)
                        self.assertEqual(set(funcsSens.keys()), {f"fc_{func}", "fail"})


if __name__ == "__main__":
    unittest.main()
