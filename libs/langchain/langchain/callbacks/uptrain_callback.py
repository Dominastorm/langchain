"""UpTrain's Callback Handler."""
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


if TYPE_CHECKING:
    import uptrain

def import_uptrain():
    try:
        import uptrain
    except ImportError as e:
        raise ImportError(
            "To use the UpTrainCallbackHandler, you need the"
            "`uptrain` package. Please install it with"
            "`pip install uptrain`.",
            e
        )

    return uptrain

class UpTrainCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to uptrain.

    Args:
        BaseCallbackHandler (_type_): _description_
    """

    REPO_URL: str = "https://github.com/uptrain-ai/uptrain"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    DOCS_URL: str = "https://docs.uptrain.ai"

    def __init__(
            self,
            checks
            ) -> None:
        """Initializes the `UpTrainCallbackHandler`."""
        super().__init__()

        uptrain = import_uptrain()

        from uptrain.framework import Settings
        import os

        # Set uptrain variables
        self.checks = checks

        # Set up uptrain settings
        self.uptrain_settings = Settings(
            uptrain_access_token=os.environ["UPTRAIN_API_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )


    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Store the prompts"""
        self.prompts = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log records to uptrain when an LLM ends."""
        from uptrain.operators import LanguageCritique, ResponseCompleteness
        import polars as pl

        print(response.generations)
        print([generation[0].text for generation in response.generations])
        print("\n\n\n\n")
        # for check in self.checks:
            # print(response.generations)
            # for i, generation in enumerate(response.generations):
            #     output = generation[0].text
            #     query = self.prompts[i]
            #     data = pl.DataFrame({"query": self.prompts, "output": [output]})
            #     if isinstance(check, LanguageCritique):
            #         operator = LanguageCritique(col_response="output")
            #         result = operator.setup(self.uptrain_settings).run(data)
            #         print(f"Language Quality: {result}")
            #     elif isinstance(check, ResponseCompleteness):
            #         operator = ResponseCompleteness(col_question="query", col_response="output")
            #         result = operator.setup(self.uptrain_settings).run(data)
            #         print(f"Response Completeness: {result}")
            #     else:
            #         raise ValueError(
            #             f"""check {check.__name__} is not supported by deepeval 
            #             callbacks."""
            #         )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing when chain starts"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain ends."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass

