
import streamlit as st


def initialize_permit_guidance():

    if "permit_round" not in st.session_state:
        st.session_state.permit_round = 1

    if "permit_answers" not in st.session_state:
        st.session_state.permit_answers = {}


def display_question_round(
    questions,
    round_number,
    receiving_water
):

    st.subheader(
        f"Permit-Writer Guidance — Round {round_number}"
    )

    st.info(
        f"Receiving Water: {receiving_water}"
    )

    st.write(
        "Answer the questions below using the information "
        "currently available. It is acceptable to identify "
        "information that still needs to be obtained."
    )

    for item in questions:

        number = item["number"]

        st.markdown(
            f"### Question {number}: {item['topic']}"
        )

        st.write(
            item["question"]
        )

        with st.expander(
            "Why is this being asked?"
        ):

            st.write(
                item["why_asked"]
            )

            st.caption(
                "EPA NPDES Permit Writers' Manual "
                f"support: PDF page {item['epa_page']}"
            )

        existing_answer = (
            st.session_state.permit_answers.get(
                number,
                ""
            )
        )

        answer = st.text_area(
            f"Response to Question {number}",
            value=existing_answer,
            height=140,
            key=f"permit_answer_{number}"
        )

        st.session_state.permit_answers[
            number
        ] = answer

        st.divider()


def permit_guidance_interface(
    receiving_water,
    first_round_questions,
    second_round_questions
):

    initialize_permit_guidance()

    st.header(
        "📝 NPDES Permit-Writer Guidance"
    )

    st.write(
        "This guided workflow combines receiving-water "
        "information from the Santa Ana Region Basin Plan "
        "with guidance from the U.S. EPA NPDES Permit "
        "Writers' Manual."
    )

    current_round = (
        st.session_state.permit_round
    )

    if current_round == 1:

        display_question_round(
            first_round_questions,
            1,
            receiving_water
        )

        if st.button(
            "Continue to Next Five Questions",
            type="primary"
        ):

            st.session_state.permit_round = 2

            st.rerun()

    elif current_round == 2:

        display_question_round(
            second_round_questions,
            2,
            receiving_water
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "← Return to First Five"
            ):

                st.session_state.permit_round = 1

                st.rerun()

        with col2:

            if st.button(
                "Save Permit Case",
                type="primary"
            ):

                st.success(
                    "Permit case responses saved "
                    "in the current session."
                )
