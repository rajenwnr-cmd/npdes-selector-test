
import streamlit as st

from permit_guidance_component import permit_guidance_interface


# ------------------------------------------------------------
# TEST DATA
# ------------------------------------------------------------

receiving_water = "Santa Ana River Reach 3"


first_round_questions = [
    {
        "number": 1,
        "topic": "Facility and Discharge Characterization",
        "question":
            "Describe the proposed discharge and facility. Include the "
            "type of facility or activity, wastewater sources, treatment "
            "processes, outfall location, average and maximum discharge "
            "flows, and pollutants known or expected to be present.",
        "why_asked":
            "EPA guidance indicates that the permit writer should understand "
            "facility operations, wastewater characteristics, treatment, "
            "flow, discharge location, and potential pollutants.",
        "epa_page": 90
    },
    {
        "number": 2,
        "topic": "Beneficial Uses and Water Quality Standards",
        "question":
            "Could the proposed discharge affect designated beneficial uses "
            "or cause or contribute to an exceedance of an applicable water "
            "quality objective?",
        "why_asked":
            "EPA guidance requires consideration of the effect of the "
            "discharge on applicable water quality standards.",
        "epa_page": 117
    },
    {
        "number": 3,
        "topic": "Pollutants of Concern and Impairments",
        "question":
            "Does the proposed discharge contain, or have the potential to "
            "contain, pollutants associated with an identified impairment?",
        "why_asked":
            "EPA guidance identifies pollutants of concern as a key first "
            "step in water quality-based permitting.",
        "epa_page": 129
    },
    {
        "number": 4,
        "topic": "Reasonable Potential and WQBELs",
        "question":
            "What effluent and receiving-water information is available to "
            "evaluate reasonable potential and the need for WQBELs?",
        "why_asked":
            "EPA guidance requires reasonable-potential evaluation before "
            "developing water quality-based effluent limitations.",
        "epa_page": 147
    },
    {
        "number": 5,
        "topic": "TMDLs and Wasteload Allocations",
        "question":
            "Is the proposed discharge subject to an applicable TMDL or "
            "wasteload allocation?",
        "why_asked":
            "Applicable wasteload allocations can affect effluent limits "
            "and permit conditions.",
        "epa_page": 130
    }
]


second_round_questions = [
    {
        "number": 6,
        "topic": "Bacteria and Impairment Follow-Up",
        "question":
            "Provide available bacteria monitoring data for the effluent.",
        "why_asked":
            "Bacteria were identified as a pollutant of concern.",
        "epa_page": 130
    },
    {
        "number": 7,
        "topic": "Effluent Data and Reasonable Potential",
        "question":
            "Summarize available effluent monitoring data for each pollutant "
            "of concern.",
        "why_asked":
            "Additional data are needed for reasonable-potential analysis.",
        "epa_page": 146
    },
    {
        "number": 8,
        "topic": "Receiving-Water Background and Critical Conditions",
        "question":
            "Provide upstream or background receiving-water data and "
            "critical-flow information.",
        "why_asked":
            "Background conditions are needed for WQBEL development.",
        "epa_page": 141
    },
    {
        "number": 9,
        "topic": "TMDL and Wasteload Allocation Verification",
        "question":
            "Confirm whether an approved TMDL establishes a wasteload "
            "allocation for the discharge.",
        "why_asked":
            "The applicable wasteload allocation has not yet been confirmed.",
        "epa_page": 129
    },
    {
        "number": 10,
        "topic": "Monitoring Requirements",
        "question":
            "Describe the existing effluent and receiving-water monitoring "
            "program.",
        "why_asked":
            "Monitoring requirements support compliance and permit decisions.",
        "epa_page": 170
    }
]


permit_guidance_interface(
    receiving_water,
    first_round_questions,
    second_round_questions
)
