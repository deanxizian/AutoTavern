import SwiftUI

struct StateReviewView: View {
    let state: WindowState?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(state?.isLive == false ? "当前演练局面" : "日志读取状态")
                .font(.system(size: 18, weight: .semibold))
            ForEach(state?.facts ?? [], id: \.label) { fact in
                LabeledContent(fact.label) {
                    Text(fact.value).multilineTextAlignment(.trailing).textSelection(.enabled)
                }
                .font(.system(size: 12))
            }
            Divider()
            if let operations = state?.recentOperations, !operations.isEmpty {
                DisclosureGroup("最近读取的操作") {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(operations) { operation in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(operation.title).font(.system(size: 13, weight: .medium))
                                Text(operation.detail).font(.system(size: 11)).foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.top, 8)
                }
                .font(.system(size: 13))
            }
            if let state, !state.cards.isEmpty {
                ForEach(cardSections) { section in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(section.title).font(.system(size: 13, weight: .semibold))
                        ForEach(section.cards) { card in
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: "rectangle.portrait").foregroundStyle(.secondary)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(card.name).font(.system(size: 14, weight: .medium))
                                    Text(card.detail).font(.system(size: 12)).foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 0)
                            }
                            .padding(.vertical, 3)
                        }
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("尚未读到卡牌").font(.system(size: 15, weight: .medium))
                    Text(state?.clientRunning == true ? "完整对局状态需要有效的对局日志。收到后会自动更新。" : "等待客户端启动并写入对局日志。")
                        .font(.system(size: 13)).foregroundStyle(.secondary).lineSpacing(4)
                }
            }
            if let state, state.isLive {
                Divider()
                ForEach(state.blockers, id: \.self) { reason in
                    Label(reason, systemImage: "clock")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var cardSections: [CardSection] {
        var sections: [CardSection] = []
        for card in state?.cards ?? [] {
            let key = card.group ?? "cards"
            if let index = sections.firstIndex(where: { $0.id == key }) {
                sections[index].cards.append(card)
            } else {
                sections.append(CardSection(id: key, title: card.groupTitle ?? "按位置核对", cards: [card]))
            }
        }
        return sections
    }
}

private struct CardSection: Identifiable {
    let id: String
    let title: String
    var cards: [VisibleCard]
}

struct HistoryView: View {
    let events: [FeedbackEvent]

    var body: some View {
        DisclosureGroup("最近反馈") {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(events) { event in
                    HStack(alignment: .top) {
                        Image(systemName: event.outcome == "executed" ? "checkmark.circle" : "minus.circle")
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(event.title).font(.system(size: 12, weight: .medium))
                            Text((event.mode == "rehearsal" ? "演练反馈 · " : "回放反馈 · ") + (event.outcome == "executed" ? "你标记了已执行" : "你标记了未执行"))
                                .font(.system(size: 11)).foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                }
            }
            .padding(.top, 12)
        }
        .font(.system(size: 12)).foregroundStyle(.secondary)
    }
}
