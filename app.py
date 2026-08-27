
import json
import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NPDES Permit Writing Assistant",
    page_icon="🌊",
    layout="wide"
)

st.title("NPDES Permit Writing Assistant")

st.write(
    """
    This development prototype identifies a receiving water
    within the Santa Ana Region and automatically retrieves
    Basin Plan information relevant to NPDES permit development.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_selector():

    return pd.read_csv(
        "region8_receiving_water_selector.csv"
    )


@st.cache_data
def load_rag_data():

    with open(
        "rag_chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    embeddings = np.load(
        "chunk_embeddings.npy"
    )

    return chunks, embeddings


@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    return embedding_model, reranker


selector = load_selector()

rag_chunks, chunk_embeddings = (
    load_rag_data()
)

embedding_model, reranker = (
    load_models()
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_profile_candidates(
    query,
    initial_k=30,
    final_k=5
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    candidate_indices = np.argsort(
        similarities
    )[::-1][:initial_k]

    candidates = []

    for idx in candidate_indices:

        item = rag_chunks[idx].copy()

        item["similarity"] = float(
            similarities[idx]
        )

        candidates.append(item)


    pairs = [
        [query, item["text"]]
        for item in candidates
    ]

    rerank_scores = reranker.predict(
        pairs
    )


    # Normalize cross-encoder scores
    rerank_scores = np.asarray(
        rerank_scores,
        dtype=float
    )

    if (
        len(rerank_scores) > 0
        and rerank_scores.max()
        != rerank_scores.min()
    ):

        rerank_norm = (
            rerank_scores
            - rerank_scores.min()
        ) / (
            rerank_scores.max()
            - rerank_scores.min()
        )

    else:

        rerank_norm = np.ones(
            len(rerank_scores)
        )


    for i, item in enumerate(candidates):

        item["rerank_score"] = float(
            rerank_scores[i]
        )

        item["final_score"] = float(
            0.70 * rerank_norm[i]
            + 0.30 * item["similarity"]
        )


    candidates.sort(
        key=lambda x:
            x["final_score"],
        reverse=True
    )

    return candidates[:final_k]


# ============================================================
# PROFILE EVIDENCE SELECTION
# ============================================================

def choose_profile_result(
    section_name,
    results,
    receiving_water
):

    if not results:
        return None


    reach_root = (
        receiving_water
        .lower()
        .split(" - ")[0]
    )

    candidates = []


    for result in results:

        item = result.copy()

        text = (
            item.get(
                "text",
                ""
            )
            .lower()
        )

        score = item.get(
            "final_score",
            0
        )

        bonus = 0.0


        if section_name == "beneficial_uses":

            if "table 3-1" in text:
                bonus += 0.35

            if reach_root in text:
                bonus += 0.30

            if "beneficial use" in text:
                bonus += 0.15


        elif section_name == "water_quality_objectives":

            if "table 4-1" in text:
                bonus += 0.35

            if "water quality objectives" in text:
                bonus += 0.20

            if reach_root in text:
                bonus += 0.35


        elif section_name == "impairments":

            if "303(d)" in text:
                bonus += 0.40

            if "impaired waters" in text:
                bonus += 0.25

            if reach_root in text:
                bonus += 0.35


        elif section_name == "tmdls":

            if "tmdl" in text:
                bonus += 0.20

            if "numeric target" in text:
                bonus += 0.30

            if "wasteload allocation" in text:
                bonus += 0.25

            if "load allocation" in text:
                bonus += 0.20

            if "table 6-1x" in text:
                bonus += 0.40


        elif section_name == "implementation":

            if reach_root in text:
                bonus += 0.35

            if "high flow suspension" in text:
                bonus += 0.30

            if "wasteload allocation" in text:
                bonus += 0.20

            if "monitoring" in text:
                bonus += 0.15


        item[
            "profile_score"
        ] = (
            score + bonus
        )

        candidates.append(
            item
        )


    candidates.sort(
        key=lambda x:
            x["profile_score"],
        reverse=True
    )

    return candidates[0]


def clean_source_name(source):

    return (
        str(source)
        .replace(".pdf", "")
        .replace("_", " ")
        .strip()
    )


def clean_evidence(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ============================================================
# BUILD COMPLETE PROFILE
# ============================================================

def build_receiving_water_profile(
    receiving_water,
    wb_refkey,
    counties
):

    tasks = OrderedDict({

        "beneficial_uses":
            f"What designated beneficial uses apply to "
            f"{receiving_water}?",

        "water_quality_objectives":
            f"What water quality objectives are relevant "
            f"to regulating a wastewater discharge to "
            f"{receiving_water}?",

        "impairments":
            f"What impairments, 303(d) listings, or "
            f"pollutants of concern are identified for "
            f"{receiving_water}?",

        "tmdls":
            f"What TMDLs apply to {receiving_water}, "
            f"including numeric targets, wasteload "
            f"allocations, and load allocations?",

        "implementation":
            f"What Basin Plan implementation requirements "
            f"or special provisions are relevant to an "
            f"NPDES discharge to {receiving_water}?"
    })


    labels = {

        "beneficial_uses":
            "Designated Beneficial Uses",

        "water_quality_objectives":
            "Water Quality Objectives",

        "impairments":
            "303(d) Impairments / Pollutants of Concern",

        "tmdls":
            "Applicable TMDLs / Allocations",

        "implementation":
            "Implementation / Special Provisions"
    }


    sections = OrderedDict()


    for section_name, query in tasks.items():

        results = retrieve_profile_candidates(
            query
        )

        best = choose_profile_result(
            section_name,
            results,
            receiving_water
        )


        if best is None:

            sections[
                section_name
            ] = {

                "label":
                    labels[
                        section_name
                    ],

                "source": None,

                "page": None,

                "evidence":
                    "No strong Basin Plan evidence "
                    "was identified."
            }

            continue


        sections[
            section_name
        ] = {

            "label":
                labels[
                    section_name
                ],

            "source":
                clean_source_name(
                    best["source"]
                ),

            "page":
                best["page"],

            "evidence":
                clean_evidence(
                    best["text"]
                )
        }


    return {

        "receiving_water":
            receiving_water,

        "WB_REFKEY":
            wb_refkey,

        "counties":
            counties,

        "sections":
            sections
    }


# ============================================================
# STEP 1 — COUNTY SELECTION
# ============================================================

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


# ============================================================
# FILTER RECEIVING WATERS
# ============================================================

if selected_counties:

    def county_match(value):

        if pd.isna(value):
            return False

        counties = [
            c.strip()
            for c in str(value).split(";")
        ]

        return any(
            county in counties
            for county in selected_counties
        )


    filtered = selector[
        selector["counties"]
        .apply(county_match)
    ].copy()

    filtered = filtered.sort_values(
        "receiving_water_reach"
    )


    # ========================================================
    # STEP 2 — REVIEW AVAILABLE RECEIVING WATERS
    # ========================================================

    st.markdown("---")

    st.header(
        "2. Review Available Receiving Waters / Reaches"
    )

    st.write(
        """
        Use the list below to review the receiving waters
        and reaches associated with the selected county or
        counties. Then choose the applicable receiving water
        or reach in Step 3.
        """
    )

    st.write(
        f"**{len(filtered)} matching receiving-water records**"
    )


    table = filtered[
        [
            "receiving_water_reach",
            "waterbody_type",
            "counties"
        ]
    ].rename(
        columns={
            "receiving_water_reach":
                "Receiving Water / Reach",

            "waterbody_type":
                "Waterbody Type",

            "counties":
                "County / Counties"
        }
    )


    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # STEP 3 — SELECT RECEIVING WATER
    # ========================================================

    st.markdown("---")

    st.header(
        "3. Select Receiving Water / Reach"
    )


    water_options = (
        filtered[
            "receiving_water_reach"
        ]
        .dropna()
        .tolist()
    )


    selected_water = st.selectbox(
        "Select the receiving water or reach:",
        options=[""] + water_options,
        format_func=lambda x:
            "Choose a receiving water / reach"
            if x == ""
            else x
    )


    if selected_water:

        selected_record = filtered[
            filtered[
                "receiving_water_reach"
            ] == selected_water
        ].iloc[0]


        # ====================================================
        # STEP 4 — CONFIRM
        # ====================================================

        st.markdown("---")

        st.header(
            "4. Selected Receiving Water"
        )

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


        # ====================================================
        # STEP 5 — BUILD PROFILE
        # ====================================================

        st.markdown("---")

        st.header(
            "5. Basin Plan Regulatory Profile"
        )


        if st.button(
            "Generate Regulatory Profile",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Searching the Santa Ana Region Basin Plan..."
            ):

                profile = (
                    build_receiving_water_profile(
                        selected_record[
                            "receiving_water_reach"
                        ],
                        selected_record[
                            "WB_REFKEY"
                        ],
                        selected_record[
                            "counties"
                        ]
                    )
                )


            st.success(
                "Regulatory profile generated."
            )


            for section in (
                profile[
                    "sections"
                ].values()
            ):

                with st.expander(
                    section["label"],
                    expanded=True
                ):

                    if section[
                        "source"
                    ] is None:

                        st.warning(
                            section[
                                "evidence"
                            ]
                        )

                    else:

                        st.markdown(
                            f"""
                            **Primary Basin Plan Source:**  
                            {section['source']}

                            **Page:**  
                            {section['page']}
                            """
                        )

                        st.markdown(
                            "**Retrieved Regulatory Evidence**"
                        )

                        st.write(
                            section[
                                "evidence"
                            ]
                        )


            st.info(
                """
                This profile identifies potentially relevant
                Basin Plan evidence for permit development.
                Retrieved information should be reviewed in
                context before establishing permit requirements
                or effluent limitations.
                """
            )


else:

    st.info(
        "Select at least one county to display "
        "available receiving waters and reaches."
    )


st.markdown("---")

st.caption(
    "Development prototype — Santa Ana Region "
    "NPDES Permit Writing Assistant."
)
