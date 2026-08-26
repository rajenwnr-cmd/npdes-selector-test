
import streamlit as st
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="NPDES Receiving Water Selector",
    page_icon="🌊",
    layout="wide"
)


# ------------------------------------------------------------
# TITLE / INTRODUCTION
# ------------------------------------------------------------

st.title("NPDES Permit Writing Assistant")

st.subheader("Select the Receiving Water")

st.write(
    """
    Select the county or counties associated with the proposed
    discharge. The system will show the Region 8 receiving waters
    and reaches associated with those counties.

    Select one receiving water or reach to begin development of
    the regulatory profile for the proposed NPDES permit.
    """
)


# ------------------------------------------------------------
# LOAD SELECTOR DATA
# ------------------------------------------------------------

csv_path = Path("region8_receiving_water_selector.csv")

if not csv_path.exists():

    st.error(
        "The receiving-water selector dataset was not found. "
        "The file region8_receiving_water_selector.csv must be "
        "stored with this Streamlit application."
    )

    st.stop()


selector = pd.read_csv(csv_path)

required_columns = [
    "counties",
    "receiving_water_reach",
    "waterbody_type",
    "WB_REFKEY"
]

missing_columns = [
    col for col in required_columns
    if col not in selector.columns
]

if missing_columns:

    st.error(
        "The selector dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ------------------------------------------------------------
# STEP 1 — COUNTY / COUNTIES
# ------------------------------------------------------------

st.markdown("---")

st.header("1. County or Counties")

county_options = [
    "Orange",
    "Riverside",
    "San Bernardino"
]

selected_counties = st.multiselect(
    "Select the county or counties associated with the receiving water:",
    options=county_options,
    placeholder="Choose one or more counties"
)


# ------------------------------------------------------------
# FILTER WATERBODIES BY COUNTY
# ------------------------------------------------------------

if selected_counties:

    def county_match(county_string):

        if pd.isna(county_string):
            return False

        record_counties = [
            c.strip()
            for c in str(county_string).split(";")
        ]

        return any(
            county in record_counties
            for county in selected_counties
        )


    filtered = selector[
        selector["counties"].apply(
            county_match
        )
    ].copy()

    filtered = (
        filtered
        .sort_values(
            "receiving_water_reach"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # STEP 2 — SHOW FILTERED RECEIVING-WATER TABLE
    # --------------------------------------------------------

    st.markdown("---")

    st.header("2. Available Receiving Waters / Reaches")

    st.write(
        f"**{len(filtered)} receiving-water records** "
        "match the selected county or counties."
    )

    display_table = filtered[
        [
            "receiving_water_reach",
            "waterbody_type",
            "counties"
        ]
    ].rename(
        columns={
            "receiving_water_reach": "Receiving Water / Reach",
            "waterbody_type": "Waterbody Type",
            "counties": "County / Counties"
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # STEP 3 — SELECT ONE RECEIVING WATER / REACH
    # --------------------------------------------------------

    st.markdown("---")

    st.header("3. Select Receiving Water / Reach")

    receiving_water_options = (
        filtered[
            "receiving_water_reach"
        ]
        .dropna()
        .tolist()
    )

    selected_water = st.selectbox(
        "Select the receiving water or reach for the proposed discharge:",
        options=[""] + receiving_water_options,
        format_func=lambda x:
            "Choose a receiving water / reach"
            if x == ""
            else x
    )


    # --------------------------------------------------------
    # STEP 4 — CONFIRM SELECTION
    # --------------------------------------------------------

    if selected_water:

        selected_record = filtered[
            filtered[
                "receiving_water_reach"
            ] == selected_water
        ].iloc[0]

        st.markdown("---")

        st.header("4. Selected Receiving Water")

        st.success(
            "Receiving water selected successfully."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                **Receiving Water / Reach**

                {selected_record['receiving_water_reach']}

                **County / Counties**

                {selected_record['counties']}
                """
            )

        with col2:

            st.markdown(
                f"""
                **Waterbody Type**

                {selected_record['waterbody_type']}

                **State Water Board Reference**

                `{selected_record['WB_REFKEY']}`
                """
            )


        # ----------------------------------------------------
        # FUTURE REGULATORY PROFILE BUTTON
        # ----------------------------------------------------

        st.markdown("---")

        if st.button(
            "Generate Regulatory Profile",
            type="primary",
            use_container_width=True
        ):

            st.info(
                """
                Regulatory profile generation will be connected
                in the next development step.

                The selected receiving water will be used to
                retrieve applicable Basin Plan beneficial uses,
                water quality objectives, impairments, TMDLs,
                and implementation provisions.
                """
            )


else:

    st.info(
        "Select at least one county to display the available "
        "Region 8 receiving waters and reaches."
    )


# ------------------------------------------------------------
# DEVELOPMENT NOTE
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Development prototype — receiving-water selector test. "
    "Regulatory information will continue to be verified against "
    "the applicable Santa Ana Region Basin Plan."
)
