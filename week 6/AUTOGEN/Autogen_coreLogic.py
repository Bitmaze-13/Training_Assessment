from  dataclasses import dataclass
from autogen_core import (
    AgentId,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler
)
import asyncio

@dataclass
class Text:
    content:str

@default_subscription
class Echo(RoutedAgent):
    def __init__(self):
        super().__init__("echo")

    @message_handler
    async def handle(self, message:Text, ctx: MessageContext) -> None:
        await self.publish_message(Text(message.content[::-1]), DefaultTopicId())

@default_subscription
class Printer(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("printer")

    @message_handler
    async def handle(self, message: Text, ctx: MessageContext) -> None:  # param must be named 'message'
        print(message.content)

async def main():
    rt = SingleThreadedAgentRuntime()
    await Echo.register(rt, "echo",lambda: Echo())
    await Printer.register(rt, "printer", lambda: Printer())
    rt.start()
    await rt.send_message(Text("DataCamp"), AgentId("echo", "default"))
    await rt.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())