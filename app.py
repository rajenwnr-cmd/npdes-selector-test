
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
    within the Santa Ana Region and retrieves Basin Plan
    information relevant to NPDES permit development.
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
# NORMALIZATION
# ============================================================

def norm_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def norm_source(source):

    if source is None:
        return ""

    source = str(source).lower()

    source = (
        source
        .replace(".pdf", "")
        .replace("_", " ")
        .replace("-", " ")
    )

    source = re.sub(
        r"\s+",
        " ",
        source
    )

    return source.strip()


def display_source_name(source):

    return (
        str(source)
        .replace(".pdf", "")
        .replace("_", " ")
        .strip()
    )


# ============================================================
# WATERBODY IDENTITY
# ============================================================

def parse_selected_waterbody(receiving_water):

    full = norm_text(
        receiving_water
    )

    short = (
        full.split(
            " - ",
            1
        )[0].strip()
        if " - " in full
        else full
    )

    reach_match = re.search(
        r"\breach\s+(\d+[a-z]?)\b",
        short,
        flags=re.I
    )

    reach_id = (
        reach_match.group(1).lower()
        if reach_match
        else None
    )

    base = re.sub(
        r"\s+reach\s+\d+[a-z]?\b.*$",
        "",
        short,
        flags=re.I
    ).strip()

    base_tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            base
        )
        if token not in {
            "river",
            "creek",
            "wash",
            "channel",
            "lake",
            "reservoir",
            "reach",
            "the",
            "of",
            "and"
        }
    ]

    return {
        "full": full,
        "short": short,
        "base": base,
        "base_tokens": base_tokens,
        "reach_id": reach_id,
        "has_formal_reach":
            reach_id is not None
    }


# ============================================================
# FORMAL REACH PROXIMITY CHECK
# ============================================================

def formal_reach_nearby(
    text,
    base,
    reach_id,
    max_gap=80
):

    text_norm = norm_text(
        text
    )

    base_norm = norm_text(
        base
    )

    if (
        not base_norm
        or not reach_id
    ):
        return False

    pattern = (
        re.escape(
            base_norm
        )
        + rf".{{0,{max_gap}}}?"
        + rf"\breach\s*[:,-]?\s*"
        + re.escape(
            reach_id
        )
        + r"\b"
    )

    return bool(
        re.search(
            pattern,
            text_norm,
            flags=re.I
        )
    )


# ============================================================
# APPLICABILITY CLASSIFIER
# ============================================================

def classify_applicability(
    text,
    identity
):

    text_norm = norm_text(
        text
    )

    full = identity[
        "full"
    ]

    base = identity[
        "base"
    ]

    reach_id = identity[
        "reach_id"
    ]

    has_reach = identity[
        "has_formal_reach"
    ]


    # --------------------------------------------------------
    # Formal numbered reach
    # --------------------------------------------------------

    if has_reach:

        if formal_reach_nearby(
            text,
            base,
            reach_id
        ):

            return {
                "status": "direct",
                "score": 1.00
            }


        if (
            base
            and base in text_norm
        ):

            return {
                "status": "context_only",
                "score": 0.25
            }


        return {
            "status": "none",
            "score": 0.00
        }


    # --------------------------------------------------------
    # Exact standalone waterbody
    # --------------------------------------------------------

    if (
        full
        and full in text_norm
    ):

        return {
            "status": "direct",
            "score": 1.00
        }


    if (
        base
        and base in text_norm
    ):

        return {
            "status": "direct",
            "score": 0.95
        }


    # --------------------------------------------------------
    # Grouped designation
    # --------------------------------------------------------

    grouped_signals = [
        "other tributaries",
        "tributaries:",
        "tributaries to",
        "other streams",
        "other creeks",
        "including",
        "and tributaries"
    ]

    grouped_context = any(
        signal in text_norm
        for signal in grouped_signals
    )


    token_matches = sum(
        token in text_norm
        for token in identity[
            "base_tokens"
        ]
    )


    if (
        grouped_context
        and len(
            identity[
                "base_tokens"
            ]
        ) > 0
        and token_matches
        == len(
            identity[
                "base_tokens"
            ]
        )
    ):

        return {
            "status": "grouped",
            "score": 0.80
        }


    # --------------------------------------------------------
    # Parent/group relationship
    # --------------------------------------------------------

    if (
        len(
            identity[
                "base_tokens"
            ]
        ) > 0
        and token_matches
        == len(
            identity[
                "base_tokens"
            ]
        )
    ):

        return {
            "status": "parent_group",
            "score": 0.60
        }


    return {
        "status": "none",
        "score": 0.00
    }


# ============================================================
# EXPECTED CHAPTERS
# ============================================================

SECTION_CHAPTERS = {

    "beneficial_uses": [
        "chapter 3"
    ],

    "water_quality_objectives": [
        "chapter 4"
    ],

    "impairments": [
        "chapter 6"
    ],

    "tmdls": [
        "chapter 6"
    ],

    "implementation": [
        "chapter 5",
        "chapter 6"
    ]
}


# ============================================================
# SAFE RETRIEVAL
# ============================================================

def retrieve_selected_waterbody(
    receiving_water,
    query,
    section_name,
    initial_k=80,
    final_k=5
):

    identity = (
        parse_selected_waterbody(
            receiving_water
        )
    )

    expected_chapters = (
        SECTION_CHAPTERS.get(
            section_name,
            []
        )
    )


    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
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

        chunk = rag_chunks[
            idx
        ].copy()

        source_norm = norm_source(
            chunk["source"]
        )


        if expected_chapters:

            if not any(
                chapter
                in source_norm
                for chapter
                in expected_chapters
            ):

                continue


        applicability = (
            classify_applicability(
                chunk["text"],
                identity
            )
        )


        chunk[
            "applicability_status"
        ] = applicability[
            "status"
        ]

        chunk[
            "applicability_score"
        ] = applicability[
            "score"
        ]

        chunk[
            "similarity"
        ] = float(
            similarities[idx]
        )

        candidates.append(
            chunk
        )


    status_priority = [
        "direct",
        "grouped",
        "parent_group",
        "context_only"
    ]


    selected_status = "none"
    selected_candidates = []


    for status in status_priority:

        matches = [
            item
            for item in candidates
            if item[
                "applicability_status"
            ] == status
        ]

        if matches:

            selected_status = status
            selected_candidates = matches
            break


    if not selected_candidates:

        return {
            "status": "none",
            "results": []
        }


    pairs = [
        [
            query,
            item["text"]
        ]
        for item
        in selected_candidates
    ]


    rerank_scores = np.asarray(
        reranker.predict(
            pairs
        ),
        dtype=float
    )


    if (
        len(
            rerank_scores
        ) > 1
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
            len(
                rerank_scores
            )
        )


    for i, item in enumerate(
        selected_candidates
    ):

        item[
            "final_score"
        ] = float(

            0.45
            * rerank_norm[i]

            + 0.30
            * item[
                "similarity"
            ]

            + 0.25
            * item[
                "applicability_score"
            ]
        )


    selected_candidates.sort(
        key=lambda x:
            x[
                "final_score"
            ],
        reverse=True
    )


    return {
        "status":
            selected_status,

        "results":
            selected_candidates[
                :final_k
            ]
    }


# ============================================================
# APPLICABILITY LABELS
# ============================================================

def applicability_label(
    status
):

    labels = {

        "direct":
            "Direct waterbody-specific evidence",

        "grouped":
            "Applicable grouped Basin Plan designation",

        "parent_group":
            "Applicable parent / tributary group evidence",

        "context_only":
            "Contextual Basin Plan evidence only",

        "none":
            "No waterbody-specific Basin Plan evidence identified"
    }

    return labels.get(
        status,
        status
    )


# ============================================================
# PLAIN-LANGUAGE SUMMARY
# ============================================================

def plain_language_summary(
    section_name,
    status,
    receiving_water,
    source,
    page
):

    water = (
        receiving_water
    )


    if status == "none":

        if section_name == "beneficial_uses":

            return (
                "The current retrieval did not identify a "
                "waterbody-specific beneficial-use provision "
                "for this selection. This does not mean that "
                "the waterbody has no designated beneficial uses; "
                "the applicable designation may occur in a grouped "
                "or parent-waterbody entry that requires further review."
            )


        if section_name == "water_quality_objectives":

            return (
                "The current retrieval did not identify a direct "
                "or grouped waterbody-specific numeric objective. "
                "Narrative objectives and other generally applicable "
                "Basin Plan requirements may still apply."
            )


        if section_name == "impairments":

            return (
                "No waterbody-specific impairment or 303(d) listing "
                "was identified in the current Basin Plan retrieval. "
                "The current Integrated Report should also be checked "
                "before making a regulatory determination."
            )


        if section_name == "tmdls":

            return (
                "No waterbody-specific TMDL was identified in the "
                "current Basin Plan retrieval. This result should not "
                "be interpreted as a definitive finding that no TMDL applies."
            )


        return (
            "No waterbody-specific implementation provision was "
            "identified in the current retrieval. General Basin Plan "
            "implementation requirements may still apply."
        )


    if status == "grouped":

        if section_name == "beneficial_uses":

            return (
                f"{water} is addressed through a grouped Basin Plan "
                f"designation rather than a standalone row. The retrieved "
                f"group entry in {source}, page {page}, explicitly includes "
                f"the selected waterbody and should be reviewed as the "
                f"applicable beneficial-use designation."
            )


        if section_name == "water_quality_objectives":

            return (
                f"{water} is included in a grouped water-quality-objective "
                f"entry. The values and conditions shown in {source}, "
                f"page {page}, should be interpreted as applying through "
                f"that grouped designation."
            )


        return (
            f"The Basin Plan addresses {water} through a grouped regulatory "
            f"entry in {source}, page {page}. The underlying group language "
            f"should be reviewed before translating it into permit requirements."
        )


    if status == "parent_group":

        return (
            f"The retrieved provision appears to apply to {water} through "
            f"a parent-waterbody or tributary-group relationship. Review "
            f"{source}, page {page}, to confirm the precise scope of applicability."
        )


    # Direct evidence
    if section_name == "beneficial_uses":

        return (
            f"The Basin Plan contains a direct beneficial-use entry for "
            f"{water}. The applicable beneficial-use symbols and footnotes "
            f"are shown in {source}, page {page}."
        )


    if section_name == "water_quality_objectives":

        return (
            f"The Basin Plan contains water-quality-objective information "
            f"directly associated with {water}. Numeric values, narrative "
            f"conditions, flow conditions, and footnotes in {source}, "
            f"page {page}, should be retained when evaluating permit requirements."
        )


    if section_name == "impairments":

        return (
            f"The Basin Plan contains impairment-related information directly "
            f"associated with {water}. The retrieved passage in {source}, "
            f"page {page}, identifies the regulatory concern that should be "
            f"considered during permit development."
        )


    if section_name == "tmdls":

        return (
            f"The Basin Plan contains TMDL-related information directly "
            f"associated with {water}. The retrieved material in {source}, "
            f"page {page}, should be reviewed for applicable numeric targets, "
            f"wasteload allocations, load allocations, and implementation provisions."
        )


    return (
        f"The Basin Plan contains implementation or special-provision language "
        f"directly associated with {water}. Review {source}, page {page}, "
        f"for conditions that may affect monitoring, implementation, or permit development."
    )


# ============================================================
# PROFILE BUILDER
# ============================================================

def build_profile(
    receiving_water,
    wb_refkey,
    counties
):

    tasks = OrderedDict({

        "beneficial_uses":
            f"What designated beneficial uses apply to "
            f"{receiving_water}?",

        "water_quality_objectives":
            f"What water quality objectives apply to "
            f"{receiving_water}?",

        "impairments":
            f"What 303(d) impairments or pollutants of concern "
            f"are identified for {receiving_water}?",

        "tmdls":
            f"What TMDLs, numeric targets, or allocations "
            f"apply to {receiving_water}?",

        "implementation":
            f"What Basin Plan implementation provisions "
            f"apply to {receiving_water}?"
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

        output = (
            retrieve_selected_waterbody(
                receiving_water,
                query,
                section_name
            )
        )


        status = output[
            "status"
        ]


        if not output[
            "results"
        ]:

            sections[
                section_name
            ] = {

                "label":
                    labels[
                        section_name
                    ],

                "status":
                    status,

                "status_label":
                    applicability_label(
                        status
                    ),

                "source":
                    None,

                "page":
                    None,

                "summary":
                    plain_language_summary(
                        section_name,
                        status,
                        receiving_water,
                        None,
                        None
                    ),

                "evidence":
                    None
            }

            continue


        best = output[
            "results"
        ][0]


        source = (
            display_source_name(
                best["source"]
            )
        )


        page = best[
            "page"
        ]


        sections[
            section_name
        ] = {

            "label":
                labels[
                    section_name
                ],

            "status":
                status,

            "status_label":
                applicability_label(
                    status
                ),

            "source":
                source,

            "page":
                page,

            "summary":
                plain_language_summary(
                    section_name,
                    status,
                    receiving_water,
                    source,
                    page
                ),

            "evidence":
                re.sub(
                    r"\s+",
                    " ",
                    str(
                        best["text"]
                    )
                ).strip()
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
# STEP 1 — COUNTY
# ============================================================

st.markdown("---")

st.header(
    "1. County or Counties"
)

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


if selected_counties:

    def county_match(
        value
    ):

        if pd.isna(
            value
        ):
            return False

        record_counties = [
            c.strip()
            for c in str(
                value
            ).split(";")
        ]

        return any(
            county
            in record_counties
            for county
            in selected_counties
        )


    filtered = selector[
        selector[
            "counties"
        ].apply(
            county_match
        )
    ].copy()


    filtered = (
        filtered
        .sort_values(
            "receiving_water_reach"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # STEP 2 — REVIEW
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


    display_table = filtered[
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
        display_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # STEP 3 — SELECT
    # ========================================================

    st.markdown("---")

    st.header(
        "3. Select Receiving Water / Reach"
    )


    options = (
        filtered[
            "receiving_water_reach"
        ]
        .dropna()
        .tolist()
    )


    selected_water = st.selectbox(
        "Select the receiving water or reach:",
        options=[""] + options,
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


        col1, col2 = st.columns(
            2
        )


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
        # STEP 5 — PROFILE
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

                profile = build_profile(
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


            st.success(
                "Regulatory profile generated."
            )


            for section in (
                profile[
                    "sections"
                ].values()
            ):

                st.subheader(
                    section[
                        "label"
                    ]
                )


                # --------------------------------------------
                # Plain-language summary
                # --------------------------------------------

                st.markdown(
                    "**Plain-language summary**"
                )

                st.write(
                    section[
                        "summary"
                    ]
                )


                # --------------------------------------------
                # Applicability
                # --------------------------------------------

                status = section[
                    "status"
                ]


                if status == "direct":

                    st.success(
                        section[
                            "status_label"
                        ]
                    )

                elif status in [
                    "grouped",
                    "parent_group"
                ]:

                    st.info(
                        section[
                            "status_label"
                        ]
                    )

                elif status == "context_only":

                    st.warning(
                        section[
                            "status_label"
                        ]
                    )

                else:

                    st.warning(
                        section[
                            "status_label"
                        ]
                    )


                # --------------------------------------------
                # Evidence
                # --------------------------------------------

                if section[
                    "source"
                ] is not None:

                    st.markdown(
                        f"""
                        **Primary Basin Plan Source:**  
                        {section['source']}

                        **Page:**  
                        {section['page']}
                        """
                    )


                    with st.expander(
                        "View retrieved regulatory text"
                    ):

                        st.write(
                            section[
                                "evidence"
                            ]
                        )


                st.markdown("---")


            st.info(
                """
                The plain-language summaries above are conservative
                interpretations of retrieved Basin Plan evidence.
                They are not final permit determinations.

                Source text should be reviewed before establishing
                effluent limitations, monitoring requirements, or
                other permit conditions.
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
